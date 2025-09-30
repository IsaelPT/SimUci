"""Instrument cluster assignment and sampled distributions for a few patients.

This script prints, for each patient (small n):
 - the feature vector used for clustering
 - distances to centroids
 - chosen centroid
 - one sample from each distribution function used by the simulation

Run deterministically with a seed so the outputs are reproducible.
"""

from pathlib import Path
from utils.helpers import get_true_data_for_validation, build_row_from_dataframe
from uci.distribuciones import (
    clustering,
    tiemp_VAM_cluster0,
    tiemp_postUCI_cluster0,
    estad_UTI_cluster0,
    tiemp_VAM_cluster1,
    tiemp_postUCI_clustet1,
    estad_UTI_cluster1,
)
from utils.constants import DFCENTROIDES_CSV_PATH
import numpy as np
import pandas as pd

OUT = Path("scripts/out_plots")
OUT.mkdir(parents=True, exist_ok=True)


def inspect(n_patients=5, seed=123):
    print(f"Running inspection for {n_patients} patients with seed={seed}")
    df = get_true_data_for_validation(seed=seed).reset_index(drop=True)

    for i in range(min(n_patients, len(df))):
        print("\n--- PATIENT", i, "---")
        # Build row dict like the simulation does
        row = build_row_from_dataframe(df, i)
        # clustering expects 11 positional args in this order:
        # (Edad, Diag_Ing1, Diag_Ing2, Diag_Ing3, Diag_Ing4, APACHE, InsufResp, va, EstadiaUTI, TiempoVAM, Est_PreUCI)
        features = (
            int(row["edad"]),
            int(row["d1"]),
            int(row["d2"]),
            int(row["d3"]),
            int(row["d4"]),
            int(row["apache"]),
            int(row["insuf"]),
            int(row["va"]),
            int(row["estuci"]),
            int(row["tiempo_vam"]),
            int(row["estpreuci"]),
        )
        print("features:", features)

        cl = clustering(*features)
        # Also print distances to each centroid and the centroid values for diagnosis
        try:
            cent_df = pd.read_csv(DFCENTROIDES_CSV_PATH)
            cent_arr = cent_df.iloc[:, 0:12].to_numpy(dtype=float)
            feat_arr = np.array(features, dtype=float).reshape(1, -1)
            diffs = cent_arr - feat_arr
            dists = np.linalg.norm(diffs, axis=1)
            print("distances_to_centroids:", np.round(dists, 3).tolist())
            print("chosen_cluster:", int(np.argmin(dists)))
            print("chosen_centroid_row:", cent_df.iloc[int(np.argmin(dists)), :].to_dict())
        except Exception as e:
            print("could not read centroids or compute distances:", e)

        # Sample a few draws from the distributions depending on cluster
        is_cluster_zero = cl == 0
        if is_cluster_zero:
            vam = tiemp_VAM_cluster0()
            post = tiemp_postUCI_cluster0()
            uci = estad_UTI_cluster0()
        else:
            vam = tiemp_VAM_cluster1()
            post = tiemp_postUCI_clustet1()
            uci = estad_UTI_cluster1()

        print("sampled (raw) VAM:", vam)
        print("sampled (raw) PostUCI:", post)
        print("sampled (raw) UCI stay:", uci)


if __name__ == "__main__":
    inspect()
