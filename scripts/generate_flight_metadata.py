import pandas as pd
from traffic.core import Traffic
from ground_analysis.extract_flight_info import extract_dep_info

if __name__ == "__main__":
    data_dir = "../data/intermediate/"
    dep_fn = "taxi_zurich_2019_takeoff.pkl"
    arr_fn = "taxi_zurich_2019_landin_v3.pkl"

    _start = pd.Timestamp("now")
    print(f"started at {_start}")

    v = 29
    """
        t_dep = (
            t_ori.iterate_lazy()  # query(f'origin=="LSZH" & destination != "LSZH"')
            .landing_at("LSZH")  # TODO: CHANGE TO takeoff_from("LSZH")
            .aircraft_data()
            .cumulative_distance()
            .eval(desc="", max_workers=23)
        ).clean_invalid()
        t_dep.to_pickle(full_path_traf)
    """
    # fdf = extract_dep_info(traf, v, fn="flight_df_DEP")

    # traf = (
    #     Traffic.from_file(data_dir + arr_fn)
    #     .aircraft_data()
    #     .cumulative_distance()
    #     .eval(desc="", max_workers=23)
    # )
    # fdf = extract_dep_info(traf, v, fn="flight_df_ARR")

    traf = (
        Traffic.from_file(data_dir + dep_fn)
        .aircraft_data()
        .cumulative_distance()
        .eval(desc="", max_workers=23)
    )
    fdf = extract_dep_info(traf, v, fn="flight_df_DEP")

    _duration = pd.Timestamp("now") - _start
    print(f"done in {_duration}")