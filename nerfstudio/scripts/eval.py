# Copyright 2026 the authors of IDSplat and contributors.
# Copyright 2025 the authors of NeuRAD and contributors.
# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import tyro

from nerfstudio.engine.trainer import TrainerConfig, detach_items_if_tensor
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # Optional path to update the data root to
    data_root_path: Optional[Path] = None
    # Optional flag to use camopt in eval
    use_camopt_in_eval: bool = False
    # Optional flag to dump images to disk
    dump_img_to_disk: bool = False
    # Optional flag to compute dynamic metrics
    dynamic_gt_masks_dir: Optional[str] = None
    # Optional flag to upload to wanbd
    upload_to_wandb: bool = False

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, step = eval_setup(
            self.load_config, update_config_callback=self.update_config
        )
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            metrics_dict = pipeline.get_average_eval_image_metrics(
                output_path=self.render_output_path, dump_img_to_disk=self.dump_img_to_disk
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        if self.upload_to_wandb:
            # Log metrics to wandb / tensorboard / comet if enabled, mirroring training behavior.
            writer_log_path = config.get_base_dir() / config.logging.relative_log_dir
            writer.setup_event_writer(
                config.is_wandb_enabled(),
                config.is_tensorboard_enabled(),
                config.is_comet_enabled(),
                log_dir=writer_log_path,
                experiment_name=config.experiment_name + "_re_eval",
                project_name=config.project_name,
            )
            writer.put_dict(
                name="Eval Images Metrics Dict (all images)",
                scalar_dict=detach_items_if_tensor(metrics_dict),
                step=step,
            )
            writer.write_out_storage()
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

    def update_config(self, config: TrainerConfig) -> TrainerConfig:
        config.pipeline.datamanager.dataparser.data = self.data_root_path
        # temp hax
        config.pipeline.model.use_camopt_in_eval = self.use_camopt_in_eval
        config.pipeline.datamanager.dataparser.dynamic_gt_masks_dir = self.dynamic_gt_masks_dir

        return config


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
