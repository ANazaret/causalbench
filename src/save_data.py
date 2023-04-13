"""
Copyright 2023  Mathieu Chevalley, Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Tuple

import numpy as np
import torch
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime


class GRNBoost(AbstractInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.n_workers = 16
        self.threads_per_worker = 8
        self.gene_expression_threshold = 0.15

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        torch.save({"expression_matrix": expression_matrix, "interventions": interventions, "gene_names": gene_names}, "k562-100.pt")
        return []
