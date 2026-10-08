# Copyright 2026 the authors of IDSplat and contributors.
# Copyright 2024 the authors of IDSplat and contributors.
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

"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Union

import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import ExternalMethodDummyTrainerConfig, get_external_methods
from nerfstudio.data.datamanagers.idsplat_datamanager import IDSplatDatamanagerConfig
from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSetDataParserConfig
from nerfstudio.data.dataparsers.wod_dataparser import WoDParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.idsplat import IDSplatModelConfig
from nerfstudio.pipelines.idsplat_pipeline import IDSplatPipelineConfig
from nerfstudio.plugins.registry import discover_methods

method_configs: Dict[str, Union[TrainerConfig, ExternalMethodDummyTrainerConfig]] = {}
descriptions = {
    "idsplat": "Instance-Decomposed Gaussian Splatting for autonomous driving",
}

method_configs["idsplat"] = TrainerConfig(
    method_name="idsplat",
    steps_per_eval_image=1000,
    steps_per_eval_batch=0,
    steps_per_save=30_000,
    steps_per_eval_all_images=15_000,
    max_num_iterations=30_001,
    mixed_precision=False,
    save_only_latest_checkpoint=False,
    pipeline=IDSplatPipelineConfig(
        calc_fid_steps=(99999999,),  # skip fid calculation
        calc_chamfer_steps=(30_000,),
        datamanager=IDSplatDatamanagerConfig(
            dataparser=PandaSetDataParserConfig(
                add_missing_points=True,
                load_cuboids=True,
                object_masks_dir="data/pandaset_grounded_tracked_masks",
            ),
            cache_images_type="uint8",
            use_voxel_based_relabeling=False,
        ),
        model=IDSplatModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-7, max_steps=30000),
        },
        "camera_velocity_opt_linear": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-6, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "camera_velocity_opt_angular": {
            "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "camera_velocity_opt_time_to_center_pixel": {
            "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-7, max_steps=30000, warmup_steps=10000, lr_pre_warmup=0
            ),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30001, warmup_steps=500),
        },
        "actor_positions": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30001, warmup_steps=1500),
        },
        "actor_rotations_6d": {
            "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=30001, warmup_steps=1500),
        },
        "actor_vel_linear": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30001, warmup_steps=2500),
        },
        "actor_vel_angular": {
            "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=30001, warmup_steps=2500),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


method_configs["idsplat-wod"] = TrainerConfig(
    method_name="idsplat-wod",
    steps_per_eval_image=1000,
    steps_per_eval_batch=0,
    steps_per_save=30_000,
    steps_per_eval_all_images=15_000,
    max_num_iterations=30_001,
    mixed_precision=False,
    save_only_latest_checkpoint=False,
    pipeline=IDSplatPipelineConfig(
        calc_fid_steps=(99999999,),
        calc_chamfer_steps=(30_000,),
        datamanager=IDSplatDatamanagerConfig(
            dataparser=WoDParserConfig(),
            cache_images_type="uint8",
            use_voxel_based_relabeling=False,
            oversample_cameras_factor=4,
            camera_res_scale_factor=0.5,
        ),
        model=IDSplatModelConfig(
            mcmc_cap_max=10_000_000,
            max_num_seed_points=5_000_000,
            n_extra_random_points=1_000_000,
            n_far_points=1_000_000,
        ),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-7, max_steps=30000),
        },
        "camera_velocity_opt_linear": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-6, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "camera_velocity_opt_angular": {
            "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "camera_velocity_opt_time_to_center_pixel": {
            "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-7, max_steps=30000, warmup_steps=10000, lr_pre_warmup=0
            ),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30001, warmup_steps=500),
        },
        "actor_positions": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30001, warmup_steps=1500),
        },
        "actor_rotations_6d": {
            "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=30001, warmup_steps=1500),
        },
        "actor_vel_linear": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30001, warmup_steps=2500),
        },
        "actor_vel_angular": {
            "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=30001, warmup_steps=2500),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
descriptions["idsplat-wod"] = "IDSplat for Waymo Open Dataset (WOD-tuned defaults)"


def merge_methods(methods, method_descriptions, new_methods, new_descriptions, overwrite=True):
    """Merge new methods and descriptions into existing methods and descriptions."""
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


all_methods, all_descriptions = method_configs, descriptions
# Add discovered external methods
all_methods, all_descriptions = merge_methods(all_methods, all_descriptions, *discover_methods())
all_methods, all_descriptions = sort_methods(all_methods, all_descriptions)

# Register all possible external methods which can be installed with Nerfstudio
all_methods, all_descriptions = merge_methods(
    all_methods, all_descriptions, *sort_methods(*get_external_methods()), overwrite=False
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
