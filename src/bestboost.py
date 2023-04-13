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


class BestBoost(AbstractInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.n_workers = 8
        self.threads_per_worker = 2
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

        # # encode each intervention label into a one-hot vector
        # import pandas as pd
        # interventions_dummy = pd.get_dummies(interventions)
        # # add suffix _intervention to each column
        # interventions_dummy.columns = [str(col) + "_intervention" for col in interventions_dummy.columns]
        # # add the one-hot vectors to the expression matrix
        # expression_matrix = np.concatenate((expression_matrix, interventions_dummy.values), axis=1)
        # # add the intervention labels to the gene names
        # gene_names = gene_names + list(interventions_dummy.columns)

        # The GRNBoost algo was tailored for only observational data.
        # You may want to modify the algo to take into account the perturbation information in the "interventions" input.
        # This may be achieved by directly modifying the algorithm or by modulating the expression matrix that is given as input.
        # change to grnboost3 for the new version
        network, importances_df = betterboost(
            expression_data=expression_matrix,
            gene_names=gene_names,
            interventions=interventions,
            client_or_address=custom_client,
            seed=seed,
            early_stop_window_length=15,
            verbose=True,
            use_interventions=True,
            # tf_names=gene_names[:10]
        )

        # You may want to postprocess the output network to select the edges with stronger expected causal effects.
        import pandas as pd

        n_interventions = len(set(interventions).intersection(gene_names))
        network_sorted_by_importance = network.sort_values(
            "importance", ascending=False
        )

        # take max importance scores from the importances.
        limit = min(1000, network_sorted_by_importance.shape[0])
        edge_mtx = network_sorted_by_importance[["TF", "target"]].values[:limit]
        edges = [(s, t) for s, t in edge_mtx]

        importances_df.to_csv(
            f"output/bestboost-importances-{n_interventions}.csv"
        )

        network.to_csv(f"output/bestboost-{n_interventions}.csv")
        torch.save(edges, f"output/bestboost-{n_interventions}-edges.pt")
        return edges


if __name__ == "__main__":
    # load data
    # filter
    n_genes = 100
    n_obs = 10000
    data = torch.load("../rpe1-25.pt")
    expression_matrix = data["expression_matrix"][0:n_obs, 0:n_genes]
    interventions = data["interventions"][0:n_obs]
    gene_names = data["gene_names"][0:n_genes]
    a = BestBoost()
    a(expression_matrix, interventions, gene_names, TrainingRegime.PartialIntervational)