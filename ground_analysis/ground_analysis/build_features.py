import pandas as pd
from pickle import load

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


from .extract_flight_info import (
    assign_asma_mode,
    assign_wtc,
    compute_avg_duration,
    count_movements_and_runway_conf,
    assign_twy_occ,
    count_wtc,
    assign_wtc_ratios,
    assign_twy_rolling_max,
    compute_aqli,
    compute_scfi,
    compute_sifi,
    onehot_col,
)
from .make_dataset import load_wtc


def assign_next_avg_delay(
    flight_df,
    avg_delay_col="avg_delay",
    dt="30T",
    name="nextAvgDelay",
    tolerance=pd.Timedelta("10T"),
):
    dt_pred = pd.Timedelta(dt)
    flight_df = flight_df.sort_values("on_runway_time")
    target_df = (
        flight_df[["on_runway_time", avg_delay_col]]
        .assign(on_runway_time=lambda x: x.on_runway_time - dt_pred)
        .rename(columns={avg_delay_col: name})
    )
    return pd.merge_asof(
        flight_df,
        target_df,
        on="on_runway_time",
        direction="nearest",
        tolerance=tolerance,
    )


def make_delay_cat(df, col, cuts=[2, 5]):
    """
    cuts should be times of separation between categories
    The list must be sorted
    """
    assert all(cuts[i] <= cuts[i + 1] for i in range(len(cuts) - 1))
    for i in range(len(cuts)):
        if i == 0:
            df.loc[df[col] <= cuts[0], col + "_cat"] = 0
        else:
            df.loc[(df[col] > cuts[i - 1]) & (df[col] <= cuts[i]), col + "_cat"] = i
    df.loc[df[col] > cuts[-1], col + "_cat"] = len(cuts)
    return df


def assign_avg_dep_speed(
    dep_arr_df,
    dt_str="30T",
    speed_col="avg_speed",
    tolerance=pd.Timedelta("2T"),
    internal_query=None,
):
    dep_arr_df_filtered = (
        dep_arr_df.query(internal_query) if internal_query is not None else dep_arr_df
    )
    new_col = f"{speed_col}_last{dt_str}"
    avg_avg_speed_df = (
        dep_arr_df_filtered.sort_values("on_runway_time")
        .set_index("on_runway_time")
        .rolling(dt_str, closed="left")
        .agg({speed_col: "mean"})
        .rename(columns={speed_col: new_col})
    )
    return pd.merge(
        dep_arr_df.sort_values("on_runway_time"),
        avg_avg_speed_df.query(f"{new_col}<10"),
        left_on="on_runway_time",
        right_index=True,
        # tolerance=tolerance,
        how="left",
    )


def assign_max_turnaround(
    dep_arr_df, dt_str="30T", tolerance=pd.Timedelta("2T"), max_turnaround="6h"
):
    dep_arr_df_filtered = dep_arr_df.query(
        f'mvt_type=="DEP" & parking_position==parking_position & turnaround<"{max_turnaround}"'
    )
    max_turnaround_df = (
        dep_arr_df_filtered.sort_values("first_movement_start")
        .set_index("first_movement_start")
        .rolling(dt_str, closed="left")
        .agg({"turnaround_minutes": "max"})
        .rename(columns={"turnaround_minutes": f"max_turnaround_minutes_last{dt_str}"})
    )
    return pd.merge_asof(
        dep_arr_df.sort_values("first_movement_start"),
        max_turnaround_df,
        left_on="first_movement_start",
        right_index=True,
        tolerance=tolerance,
    )


def assign_mean_sifis(
    dep_arr_df, sifi_cols, dt_str="30T", tolerance=pd.Timedelta("2T")
):
    """
    For each sifi col in the list sifi_cols compute the rolling average
    over the last 30 min for each reference departure
    """
    dep_arr_df_filtered = dep_arr_df.query(
        'mvt_type=="DEP" & parking_position==parking_position'
    )
    # We sort by runway arrival time as these variables
    # are based on completed taxi phase
    for col in sifi_cols:
        mean_sifi_df = (
            dep_arr_df_filtered.sort_values("on_runway_time")
            .set_index("on_runway_time")
            .rolling(dt_str, closed="left")
            .agg({col: "mean"})
            .rename(columns={col: f"mean_{col}_last{dt_str}"})
        )
        dep_arr_df = pd.merge_asof(
            dep_arr_df.sort_values("on_runway_time"),
            mean_sifi_df,
            left_on="on_runway_time",
            right_index=True,
            tolerance=tolerance,
        )
    return dep_arr_df


def assign_atmap_weather_score(flight_df, atmap_cols=None):
    """
    Suppose that the dataframe given has atmap score for each category
    Assign the mean value between those categories to a new column 'atmap_score'
    """
    if atmap_cols is None:
        atmap_cols = [
            "atmap_visibility",
            "atmap_wind",
            "atmap_precipitation",
            "atmap_freezing",
            "atmap_danger",
        ]
    flight_df["atmap_score"] = flight_df[atmap_cols].agg("sum", axis=1)
    flight_df["atmap_score"] = flight_df["atmap_score"] / len(atmap_cols)
    return flight_df


def pca_on_twy(flight_df, twy_cols, n_components=1):
    """Perform a PCA on all taxiways to get more relevant features"""
    s = MinMaxScaler()
    x = s.fit_transform(flight_df[twy_cols])
    p = PCA(n_components=n_components).fit(x)
    new_cols = [f"twy_pca_{i}" for i in range(n_components)]
    flight_df[new_cols] = p.transform(flight_df[twy_cols])
    return flight_df


def gather_data(
    dep_df,
    arr_df,
    metar_df,
    atmap_df,
    asma_stats_df,
    wtc_path,
    turnaround_df,
    twy_occ_path,
):
    with open(twy_occ_path, "rb") as f:
        d_twy = load(f)

    # TODO: remove
    if "mvt_type" not in dep_df.columns:
        dep_df["mvt_type"] = "DEP"

    ################# ARRIVALS SPECIFIC OPERATIONS #############

    # Assigning text description of cluster (busy, disruted, nominal)
    modes_df = assign_asma_mode(asma_stats_df)

    # One hot encode 'mode' and compute the count for each half hour
    modes_df.loc[:, ["busy", "disrupted", "nominal"]] = pd.get_dummies(modes_df["mode"])
    modes_df = (
        modes_df.set_index("landing")
        .sort_index()
        .rolling("30T", closed="left")
        .agg({"nominal": "sum", "busy": "sum", "disrupted": "sum"})
    )

    modes_df.loc[:, "not_nom_arrival_ratio"] = (
        modes_df["busy"] + modes_df["disrupted"]
    ) / (modes_df["busy"] + modes_df["disrupted"] + modes_df["nominal"]).fillna(0)

    ################# DEPARTURES SPECIFIC OPERATIONS #############

    query_incomplete_flights = "mvt_type=='DEP' & (parking_position == parking_position | (taxi_dist>1100 & runway=='32') |  (taxi_dist>1100 & runway=='28') | (taxi_dist>2200 & runway=='16'))"

    # Computing average delays on last 30 minutes
    dep_df = compute_avg_duration(
        dep_df,
        dt="30T",
        on_col="taxi_holding_time_minutes",
        col_name="avg_delay_30min",
        internal_query=query_incomplete_flights,
    )
    dep_df = compute_avg_duration(
        dep_df,
        dt="30T",
        on_col="taxi_holding_time_minutes",
        col_name="avg_delay_30min_all_trajs",
        internal_query="mvt_type=='DEP'",
    )

    # target cols
    dep_df = assign_next_avg_delay(
        dep_df,
        avg_delay_col="avg_delay_30min",
        dt="30T",
        name="avg_delay_in30min",
    )
    dep_df = assign_next_avg_delay(
        dep_df,
        avg_delay_col="avg_delay_30min",
        dt="60T",
        name="avg_delay_in60min",
    )
    dep_df = assign_next_avg_delay(
        dep_df,
        avg_delay_col="avg_delay_30min",
        dt="90T",
        name="avg_delay_in90min",
    )
    # Categorize average delays
    dep_df = make_delay_cat(dep_df, "avg_delay_in30min", cuts=[4, 6.5])
    dep_df = make_delay_cat(dep_df, "avg_delay_in60min", cuts=[4, 6.5])
    dep_df = make_delay_cat(dep_df, "avg_delay_in90min", cuts=[4, 6.5])

    # Assigning UTC for atmap time values
    atmap_df["time_utc"] = atmap_df["time_utc"].dt.tz_localize("utc")

    # assign turnaround times
    dep_df = (
        dep_df.merge(
            turnaround_df.assign(hh=lambda x: x.start.dt.floor("30T")).drop(
                ["callsign_in", "ALDT", "AIBT", "AOBT", "start", "firstseen_min"],
                axis=1,
            ),
            how="left",
            left_on=["callsign", "icao24", "hh"],
            right_on=["callsign_out", "icao24", "hh"],
        )
        .assign(turnaround_minutes=lambda x: x.turnaround.dt.total_seconds() / 60)
        .drop("callsign_out", axis=1)
    )
    dep_df["turnaround_minutes"] = dep_df["turnaround_minutes"].fillna(0)

    # Compute average of turnaround times lower than 6h
    dep_df = assign_max_turnaround(dep_df)

    # Compute average of the mean taxi speed of departures
    dep_df = assign_avg_dep_speed(
        dep_df,
        internal_query=query_incomplete_flights,
    )
    dep_df = assign_avg_dep_speed(
        dep_df,
        dt_str="15T",
        internal_query=query_incomplete_flights,
    )

    ################# BOTH #############

    # concatenate dep and arr dataframe
    dep_arr_df = (
        pd.concat([dep_df, arr_df], ignore_index=True)
        .reset_index(drop=True)
        .sort_values("first_movement_start")
    )

    # Arrival modes
    dep_arr_df = pd.merge_asof(
        dep_arr_df, modes_df, left_on="first_movement_start", right_index=True
    )

    # Adding metar data to the global dataframe
    metar_df = metar_df.assign(hh=lambda x: x.date.dt.round("30T"))
    # Removing duplicated metar messages
    metar_df = metar_df.loc[~metar_df.hh.duplicated(keep=False)]
    dep_arr_df = dep_arr_df.merge(
        metar_df[["press", "temp", "hh"]], how="left", on="hh"
    )

    # Adding Atmap score
    atmap_cols = [
        "atmap_visibility",
        "atmap_wind",
        "atmap_precipitation",
        "atmap_freezing",
        "atmap_danger",
    ]
    atmap_df = atmap_df[atmap_cols + ["time_utc"]].assign(
        hh=lambda x: x.time_utc.dt.round("30T")
    )
    dep_arr_df = dep_arr_df.merge(
        atmap_df.drop("time_utc", axis=1), how="left", on="hh"
    )
    # then compute average of these 5 atmap categories
    dep_arr_df = assign_atmap_weather_score(dep_arr_df)

    # Operation over the entire dataframe
    dep_arr_df = count_movements_and_runway_conf(dep_arr_df)
    wtc_df = load_wtc(wtc_path)
    dep_arr_df = assign_wtc(dep_arr_df, wtc_df)
    dep_arr_df = count_wtc(dep_arr_df)
    dep_arr_df = assign_wtc_ratios(dep_arr_df)

    # Assign twy occupancy to each flight
    dep_arr_df = assign_twy_occ(dep_arr_df, d_twy)
    twy_cols = [(twyname + "_occ") for twyname in d_twy.keys()]
    dep_arr_df = assign_twy_rolling_max(dep_arr_df, twy_cols, dt_str="15T")
    dep_arr_df = pca_on_twy(dep_arr_df, twy_cols, n_components=3)

    # Assign sifi, aqli, sfci, (Yin ete al. 2018)
    # Compute NDepDep (number of departures while reference aircraft is pushback)
    dep_arr_df = compute_sifi(dep_arr_df, col_name="sifi_DEP", mvt_type="DEP")
    dep_arr_df = compute_sifi(dep_arr_df, col_name="sifi_ARR", mvt_type="ARR")
    dep_arr_df = compute_aqli(dep_arr_df, mvt_type="DEP")
    dep_arr_df = compute_scfi(dep_arr_df, mvt_type="DEP")
    dep_arr_df = compute_aqli(dep_arr_df, mvt_type="ARR")
    dep_arr_df = compute_scfi(dep_arr_df, mvt_type="ARR")

    # Rolling average of these values
    cols = ["sifi_DEP", "sifi_ARR", "aqli_DEP", "scfi_DEP", "aqli_ARR", "scfi_ARR"]
    dep_arr_df = assign_mean_sifis(dep_arr_df, cols)

    # One hot encode runway
    dep_arr_df = onehot_col(dep_arr_df, "runway", drop_col=False)
    # same for runway configuration of last 30 min
    dep_arr_df = onehot_col(dep_arr_df, "config", drop_col=False)

    # Categorize add txot
    dep_arr_df = make_delay_cat(dep_arr_df, "taxi_holding_time_minutes", cuts=[4, 8])

    # TODO: Fix turnaround affectation and nb movement computation
    # it leads to several rows for the same flight
    dep_arr_df = dep_arr_df.drop_duplicates(subset=["flight_id", "on_runway_time"])

    return dep_arr_df
