import pandas as pd
from traffic.core import Traffic
from ground_analysis.flight_processing import assign_twy_traf_para
from ground_analysis.airport_processing import get_long_twy

from pickle import load, dump


def assign_twy(f_arr_path, f_dep_path):
    """
    Assign taxiways, intended to be use on ground traffics
    """

    long_twy_df = get_long_twy("LSZH")

    for path in [f_arr_path, f_dep_path]:
        print("starting for " + path)
        traf = Traffic.from_file(path)
        traf = assign_twy_traf_para(traf, "LSZH", long_twy_df, num_processes=15)
        n = len(path) - 4  # to insert before '.pkl'
        traf.to_pickle(path[:n] + "_TWY_" + path[n:])

    print("done and saved")


def generate_twy_occ():
    f_arr_path = "../data/intermediate/taxi_zurich_2019_landin_v3_TWY_.pkl"
    f_dep_path = "../data/intermediate/taxi_zurich_2019_takeoff_TWY_.pkl"
    t_arr = Traffic.from_file(f_arr_path)
    t_dep = Traffic.from_file(f_dep_path)
    traf = t_arr + t_dep

    twy_names = traf.query("twy==twy").data.twy.unique()
    d_twy = {n: pd.DataFrame(columns=["start", "stop"]) for n in twy_names}

    for f in traf:
        if not f.data.twy.isna().all():
            df = f.onground().query("twy==twy").data
            # counting number of occurences for each twy and time spent on each
            twys_times = (
                df.groupby((df.twy != df.twy.shift()).cumsum())
                .agg({"timestamp": ["min", "max"], "twy": "max"})
                .assign(dur=lambda x: x[("timestamp", "max")] - x[("timestamp", "min")])
            )  # .reset_index()
            # if duplicated taxiways, select only the one with the longest duration spent on it
            twys_times = twys_times.sort_values("dur").drop_duplicates(
                subset=("twy", "max"), keep="last"
            )
            for _, r in twys_times.iterrows():
                twy = r[("twy", "max")]
                d_twy[twy] = d_twy[twy].append(
                    {"start": r[("timestamp", "min")], "stop": r[("timestamp", "max")]},
                    ignore_index=True,
                )
    return d_twy


if __name__ == "__main__":
    _start = pd.Timestamp("now")
    d_twy = generate_twy_occ()
    print(d_twy)
    with open("../data/intermediate/twy_occ_dictionnary_py39.pkl", "wb") as f:
        dump(d_twy, f)
    _duration = pd.Timestamp("now") - _start
    print(f"done in {_duration}")
