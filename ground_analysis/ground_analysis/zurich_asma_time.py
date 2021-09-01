from __future__ import annotations

from typing import Iterator, cast

import numpy as np
import pandas as pd
from cartopy.crs import EuroPP  # type: ignore
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from traffic.core import Traffic
from traffic.data.datasets import landing_zurich_2019


def generate_clusters(
    runway_14: Traffic, nb_components: dict[str, int]
) -> Iterator[pd.DataFrame]:
    for flow, components in nb_components.items():

        current_flow = cast(Traffic, runway_14.query(f"initial_flow == '{flow}'"))

        s = MinMaxScaler()
        x = s.fit_transform(
            np.stack(
                list(
                    flight.data[["x", "y", "track_unwrapped"]].values.ravel()
                    for flight in current_flow
                )
            )
        )

        p = PCA().fit(x)
        d = DBSCAN(eps=0.2, min_samples=10)
        d.fit_predict(p.transform(x)[:, :components])

        yield pd.DataFrame.from_records(
            list(
                {
                    "flight_id": f.flight_id,
                    "start": f.start,
                    "initial_flow": flow,
                    "cluster": cluster,
                    "duration": f.duration.total_seconds() / 60,
                    # fl.stop if (fl := f.aligned_on_runway('LSZH').all()) is not None else None,
                    "landing": f.stop,
                }
                for f, cluster in zip(current_flow, d.labels_)
            )
        )


def main():

    runway_14 = (
        landing_zurich_2019.query('runway == "14"')
        .query("track == track")
        .unwrap()
        .compute_xy(EuroPP())
        .resample(50)
        .eval(desc="", max_workers=23)
    )

    nb_components = {
        "24-72": 3,
        "90-132": 2,
        "162-216": 2,
        "240-276": 2,
        "312-354": 2,
    }

    pd.concat(
        list(df for df in generate_clusters(runway_14, nb_components))
    ).to_pickle(  # to_csv("zurich_asma_stats.csv")
        "../../data/zurich_asma_stats3.pkl"
    )


if __name__ == "__main__":
    main()
