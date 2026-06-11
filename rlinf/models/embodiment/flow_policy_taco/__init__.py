# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL wrapper around the external ``flow_policy`` package (point-cloud
flow-matching policy for the TACO bimanual-Allegro task).

Requires ``flow_policy`` to be importable (editable install from
/root/flow-policy or equivalent).
"""

from omegaconf import DictConfig

from rlinf.models.embodiment.flow_policy_taco.flow_taco_policy import (
    FlowPolicyTacoForRL,
)

__all__ = ["FlowPolicyTacoForRL", "get_model"]


def get_model(cfg: DictConfig, torch_dtype=None) -> FlowPolicyTacoForRL:
    """Model builder registered under model_type='flow_policy_taco'."""
    return FlowPolicyTacoForRL.from_config(cfg)
