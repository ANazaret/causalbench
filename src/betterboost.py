from typing import List, Tuple

import distributed
import numpy as np
import torch
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import remove_lowly_expressed_genes

import sys, os
print(sys.path)
sys.path.append(os.getcwd())
print(os.getcwd())
from src.betterboost_core import betterboost


class BetterBoost(AbstractInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.n_workers = 20
        self.threads_per_worker = 4
        self.gene_expression_threshold = 0.15

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        """
            expression_matrix: numpy array of size n_samples x n_genes, which contains the expression values
                                of each gene in different cells
            interventions: list of size n_samples. Indicates which gene has been perturbed in each sample.
                            If value is "non-targeting", no gene was targeted (observational sample).
                            If value is "excluded", a gene was perturbed which is not in gene_names (a confounder was perturbed).
                            You may want to exclude those samples or still try to leverage them.
            gene_names: names of the genes of size n_genes. To be used as node names for the output graph.


        Returns:
            List of string tuples: output graph as list of edges.
        """
        # We remove genes that have a non-zero expression in less than 15% of samples.
        # You may want to select the genes differently.
        # You could also preprocess the expression matrix, for example to impute 0.0 expression values.
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix,
            gene_names,
            expression_threshold=self.gene_expression_threshold,
        )
        local_cluster = distributed.LocalCluster(
            n_workers=self.n_workers, threads_per_worker=self.threads_per_worker
        )
        custom_client = distributed.Client(local_cluster)

        # The GRNBoost algo was tailored for only observational data.
        # You may want to modify the algo to take into account the perturbation information in the "interventions" input.
        # This may be achieved by directly modifying the algorithm or by modulating the expression matrix that is given as input.
        network = betterboost(
            expression_data=expression_matrix,
            gene_names=gene_names,
            interventions=interventions,
            client_or_address=custom_client,
            seed=seed,
            early_stop_window_length=15,
            verbose=True,
            # tf_names=gene_names[:10]
        )

        # we adapted GRNBoost to use interventional data
        network = network.sort_values("pvalue")
        # remove pvalues that are above the benjamini-hochberg threshold
        n_pvalues = network["pvalue"].count()
        network["pvalue_rank"] = range(1, network.shape[0] + 1)
        network["pvalue_rank"] = network["pvalue_rank"] / n_pvalues * 0.05
        network = network[(network["pvalue"] < network["pvalue_rank"]) | network["pvalue"].isna()]

        network_sorted_by_pvalue_importance = network.sort_values(["pvalue", "importance"], ascending=[True, False])

        # get top 1000 edges
        edges = network_sorted_by_pvalue_importance[["TF", "target"]].values[0:1000]
        edges = [tuple(edge) for edge in edges]
        return edges


if __name__ == "__main__":
    # load data
    # filter
    n_genes = 100
    n_obs = 1000
    data = torch.load("../rpe1-25.pt")
    expression_matrix = data["expression_matrix"][0:n_obs, 0:n_genes]
    interventions = data["interventions"][0:n_obs]
    gene_names = data["gene_names"][0:n_genes]
    a = BetterBoost()
    a(expression_matrix, interventions, gene_names, TrainingRegime.PartialIntervational)
