import concurrent.futures
import multiprocessing
from itertools import combinations, repeat

import numpy as np
import pandas as pd
from pyproj.exceptions import CRSError
from shapely.geometry import LineString, MultiLineString, Point
from traffic.algorithms.douglas_peucker import douglas_peucker
from traffic.data import airports


def get_long_twy(airport):
    return (
        airports[airport]
        .taxiway.query("ref==ref")
        ._data.query("ref.str.len()==1")
        .groupby("ref")
        .agg({"geometry": list})["geometry"]
        .apply(MultiLineString)
        .to_frame()
    )


def get_intersections(twy_df):
    intersection_df = pd.DataFrame(columns=["twy1", "twy2", "geometry"])
    for (m, n) in combinations(twy_df.index, 2):
        twy1 = twy_df.loc[m]["geometry"]
        twy2 = twy_df.loc[n]["geometry"]
        intersec = twy1.intersects(twy2)
        if intersec:
            intersection = twy1.intersection(twy2)
            if isinstance(
                intersection, Point
            ):  # if there is multiple intersections betwwen those 2 taxys
                twy1_dist = twy1.project(intersection, normalized=True)
                twy2_dist = twy2.project(intersection, normalized=True)
                intersection_df = intersection_df.append(
                    {
                        "name": f"{m}_{n}",
                        "twy1": m,
                        "twy2": n,
                        "geometry": intersection,
                        "latitude": intersection.y,
                        "longitude": intersection.x,
                        "twy1_dist": twy1_dist,
                        "twy2_dist": twy2_dist,
                    },
                    ignore_index=True,
                )
            else:
                for i, p in enumerate(intersection):
                    twy1_dist = twy1.project(p, normalized=True)
                    twy2_dist = twy2.project(p, normalized=True)
                    intersection_df = intersection_df.append(
                        {
                            "name": f"{m}_{n}_{i}",
                            "twy1": m,
                            "twy2": n,
                            "geometry": p,
                            "latitude": p.y,
                            "longitude": p.x,
                            "twy1_dist": twy1_dist,
                            "twy2_dist": twy2_dist,
                        },
                        ignore_index=True,
                    )
    return intersection_df.set_index("name")
