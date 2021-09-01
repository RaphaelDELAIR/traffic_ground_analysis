import pandas as pd
from ground_analysis.build_features import gather_data

if __name__ == "__main__":

    fdf_dep_path = "../data/intermediate/flight_df_DEP_v29_2019-10-01 03:58:59+00:00_2019-11-30 22:14:59+00:00.pkl"
    fdf_arr_path = "../data/intermediate/flight_df_ARR_v28_2019-10-01 04:03:04+00:00_2019-11-30 21:50:22+00:00.pkl.parquet.gzip"
    metar_df_path = (
        "../data/external/metar_df2019-10-01 00:00:00_2019-12-01 00:00:00.pkl"
    )
    atmap_df_path = "../data/external/lszh_metar_atmap.csv"
    asma_stats_path = "../data/intermediate/zurich_asma_stats3.pkl"
    wtc_path = "../data/external/UK_wtc.csv"
    turnaround_path = "../data/intermediate/zurich_turnaround.pkl"
    d_twy_path = "../data/intermediate/twy_occ_dictionnary.pkl"

    dep_df = pd.read_pickle(fdf_dep_path)
    arr_df = pd.read_parquet(fdf_arr_path)
    metar_df = pd.read_pickle(metar_df_path)
    atmap_df = pd.read_csv(atmap_df_path, parse_dates=["time_utc"])
    asma_stats_df = pd.read_pickle(asma_stats_path)
    turnaround_df = pd.read_pickle(turnaround_path)

    # TODO: change, load data outside function and before its call
    dep_arr_df = gather_data(
        dep_df,
        arr_df,
        metar_df,
        atmap_df,
        asma_stats_df,
        wtc_path,
        turnaround_df,
        d_twy_path,
    )
    dep_arr_df.to_pickle(
        "../data/processed/dep_arr_df14_nextavgdelaybetteraffected.pkl"
    )
