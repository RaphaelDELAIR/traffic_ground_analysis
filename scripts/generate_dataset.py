#!/usr/bin/env python3

from make_dataset import load_extract_landing_ground_traffic_para
import pandas as pd

if __name__ == "__main__":
    from_date = pd.Timestamp("2019-10-01 00:00:00+00:00")
    to_date = pd.Timestamp("2019-12-01 00:00:00+00:00")

    _start_time = pd.Timestamp("now")
    print("Starting at " + _start_time)

    landings = load_extract_landing_ground_traffic_para(
        from_date,
        to_date,
        num_processes=None,
    )

    fn_savedfile = "taxi_zurich_2019_landin_v3.pkl"
    data_dir = "../data/intermediate/"
    landings.to_pickle(data_dir + fn_savedfile)

    _duration = pd.Timestamp("now") - _start_time

    print(f"Generated in {_duration}")
    print(f"Saved at " + data_dir + fn_savedfile)

    # NOT FOR DEPARTURES BECAUSE ALREADY GENERATED
