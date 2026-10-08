# Copyright 2026 the authors of IDSplat and contributors.
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
import torch

from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.cnns import BasicBlock

# need following import for background color override


class RGBDecoderMLP(torch.nn.Module):
    def __init__(
        self,
        in_dim=6,
        out_dim=6,
        num_layers=2,
        skip_dim=3,
        weight_init_scale=1e-2,
        hidden_dim=32,
        kernel_size=3,
        num_hidden_blocks=1,
        bias=False,
        implementation="tcnn",
    ):
        super().__init__()

        self.net = MLP(in_dim, num_layers, hidden_dim, out_dim, implementation=implementation)

        self.skip_dim = skip_dim
        self.out_dim = out_dim
        if implementation == "tcnn" and self.net_type == "mlp":
            # pad number of parameters in the last layer to a multiple of 16
            n_params_last_layer = ((out_dim + 16 - 1) // 16) * 16 * hidden_dim
            self.net.tcnn_encoding.params.data[-n_params_last_layer:] *= weight_init_scale

    def forward(self, features, ray_dirs):
        features = features.view(1, *features.shape[-3:])
        og_shape = features.shape
        albedo, spec = features.split([self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1)

        spec = torch.cat([spec, ray_dirs], dim=-1)
        spec = spec.reshape(-1, spec.shape[-1])
        spec = self.net(spec)
        spec = spec.reshape(*og_shape[:-1], self.out_dim)

        return albedo * (1 + spec[..., :3]) + spec[..., 3:]


class RGBDecoderCNN(torch.nn.Module):
    def __init__(
        self,
        in_dim=6,
        out_dim=6,
        skip_dim=3,
        weight_init_scale=1e-2,
        hidden_dim=32,
        kernel_size=3,
        num_hidden_blocks=1,
    ):
        super().__init__()
        last_layer = torch.nn.Conv2d(hidden_dim, out_dim, 1)
        last_layer.weight.data *= weight_init_scale
        layers = [BasicBlock(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2, use_bn=False)]
        for _ in range(num_hidden_blocks):
            layers.append(BasicBlock(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, use_bn=False))
        layers.append(last_layer)
        self.net = torch.nn.Sequential(*layers)
        self.skip_dim = skip_dim
        self.out_dim = out_dim

    def forward(self, features, ray_dirs):
        features = features.view(1, *features.shape[-3:])
        albedo, spec = features.split([self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1)

        spec = torch.cat([spec, ray_dirs], dim=-1)
        spec = spec.permute(0, 3, 1, 2)
        spec = self.net(spec)

        spec = spec.permute(0, 2, 3, 1)

        return albedo * (1 + spec[..., :3]) + spec[..., 3:]
