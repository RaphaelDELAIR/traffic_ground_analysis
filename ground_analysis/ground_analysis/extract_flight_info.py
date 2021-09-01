from traffic.core import Traffic
import traffic.core.geodesy as geo
from traffic.data import airports
from traffic.core.geodesy import distance
from traffic.algorithms.douglas_peucker import douglas_peucker

from pyproj.exceptions import CRSError
from shapely.geometry import Point, LineString, MultiLineString

import concurrent.futures
import multiprocessing
from tqdm import tqdm
import os


import pandas as pd
from itertools import repeat, combinations
import numpy as np
from collections import namedtuple

from .flight_processing import angle_sum, ground_movement_type, cumdist_ground

"""
This file takes intermediate data and transforms it into 
"""


ResRow = namedtuple(
    "ResRow",
    "flight_id on_runway_time taxi_holding_time rwy_holding_time total_holding_time taxi_holding_time_minutes first_movement callsign registration typecode icao24 firstseen_min start stop duration cumdist_max parking_position parking_position_duration runway pb_duration end_pb duration_minutes parking_position_duration_minutes pb_duration_minutes first_movement_start taxi_dist total_holding_time_minutes real_dur real_dur_minutes airline hh hh_num angle_sum avg_speed mvt_type taxiing_stop",
)


def process_flight(f):
    # TODO: Put 1 col per twy and binary variable with 1 or 0 if used or not OR max congestion of each twy during flight pb
    # use aircraft_data() and cumulative_distance() on traffic file first
    # Selecting right part of the flight
    if f is None:
        return None
    mvt_type = ground_movement_type(f, "LSZH")
    within_airport = f.inside_bbox(airports["LSZH"])
    try:
        moving = f.moving().onground()
    except AttributeError:
        return None

    # removing flight if its data will not provide much information
    if (
        f.is_from_inertial()
        or len(f.data) < 5
        # or f.airborne() is None
        or within_airport is None
        or moving is None
        or len(moving) < 3
        or mvt_type == "BOTH"
        or mvt_type is None
    ):
        return None

    # Start of the movement (more for departures)
    first_movement = moving.start
    first_movement_start = (
        first_movement if first_movement == first_movement else f.start
    )

    # Parking position affectation
    try:
        pp_ = within_airport.on_parking_position(airports["LSZH"]).next()
    except Exception as error:
        pp_ = None
    pp = pp_.parking_position_max if pp_ is not None else None
    pp_dur = pp_.duration if pp_ is not None else None

    # Pushback detection (for departures, will be none for arrivals)
    try:
        pb = f.pushback(airport="LSZH")
    except Exception as error:
        print(error)
        pb = None
    end_pb_time = pb.stop if pb is not None else None
    pb_time = end_pb_time - first_movement if pb is not None else None
    after_pb = moving.after(end_pb_time) if end_pb_time is not None else None

    # Stop segments
    _taxiing = after_pb if after_pb is not None else moving
    taxi_holding_time = sum(
        (slow_segment.duration for slow_segment in _taxiing.slow_taxi()),
        (pd.Timedelta(0)),
    )
    rwy_holding_time = None
    total_holding_time = sum(
        (slow_segment.duration for slow_segment in moving.slow_taxi()),
        (pd.Timedelta(0)),
    )
    taxi_holding_time_minutes = taxi_holding_time.total_seconds() / 60
    total_holding_time_minutes = total_holding_time.total_seconds() / 60

    # Name of the runway + taxiing segment
    if mvt_type == "DEP":
        if f.takeoff_from_runway("LSZH").has():
            to_rwy = f.takeoff_from_runway("LSZH").next().runway_max
            ort = f.takeoff_from_runway("LSZH").next().start
            taxiing = moving.before(ort)
        else:
            return None
    # We don't need to check other values becausde they returned a None value at the beginnning of the function
    # So this is for the case of an arrival
    if mvt_type == "ARR":
        if f.aligned_on_ils("LSZH").has():
            to_rwy = f.aligned_on_ils("LSZH").max().ILS_max
            ort = f.aligned_on_ils("LSZH").max().stop
            taxiing = moving.after(ort)
        else:
            return None

    if taxiing is None or len(taxiing) < 2:
        return None

    real_dur_minutes = taxiing.duration.total_seconds() / 60
    taxi_dist = cumdist_ground(
        taxiing
    )  # taxiing.cumulative_distance().cumdist_max * 1852
    sum_turn_angles = angle_sum(taxiing)

    hh = first_movement.floor("30T")
    hh_num = hh.hour + hh.minute / 60

    pb_duration_minutes = pb_time.total_seconds() / 60 if pb_time is not None else None

    res_tuple = ResRow(
        flight_id=f.flight_id,
        on_runway_time=ort,
        taxi_holding_time=taxi_holding_time,
        rwy_holding_time=rwy_holding_time,
        total_holding_time=total_holding_time,
        taxi_holding_time_minutes=taxi_holding_time_minutes,
        first_movement=first_movement,
        callsign=f.callsign,
        registration=f.registration,
        typecode=f.typecode,
        icao24=f.icao24,
        firstseen_min=f.firstseen_min,
        start=f.start,
        stop=f.stop,
        duration=f.duration,
        duration_minutes=f.duration.total_seconds() / 60,
        cumdist_max=f.cumdist_max,
        parking_position=pp,
        parking_position_duration=pp_dur,
        parking_position_duration_minutes=pp_dur.total_seconds() / 60
        if pp_dur is not None
        else None,
        runway=to_rwy,
        pb_duration=pb_time,
        end_pb=end_pb_time,
        pb_duration_minutes=pb_duration_minutes,
        first_movement_start=first_movement_start,
        taxi_dist=taxi_dist,
        total_holding_time_minutes=total_holding_time_minutes,
        real_dur=taxiing.duration,
        real_dur_minutes=real_dur_minutes,
        airline="".join([i for i in f.callsign if not i.isdigit()][:3]),
        hh=hh,
        hh_num=hh_num,
        angle_sum=sum_turn_angles,
        avg_speed=taxi_dist / (real_dur_minutes * 60),
        mvt_type=mvt_type,
        taxiing_stop=taxiing.stop,
    )
    return res_tuple


def extract_dep_info(
    t_ori, v, num_processes=None, dir_path="../data/intermediate/", fn="flight_df"
):
    """
    assign_id(). if not already set for landings
    aircraft_data().cumulative_distance() should have been applied to input traffic
    """
    start_str = str(t_ori.start_time)
    stop_str = str(t_ori.end_time)

    name_file = f"{fn}_v{v}_{start_str}_{stop_str}.pkl"
    full_path = dir_path + name_file

    if os.path.exists(full_path):
        print(f"Already found a pickle file for this range of dates : {full_path}")
        return pd.read_pickle(full_path)

    _start = pd.Timestamp("now")

    if num_processes == None:
        num_processes = multiprocessing.cpu_count()

    print(f"Launching on {num_processes} cores")

    l_res_tuple = []

    with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
        for res_tuple in tqdm(pool.map(process_flight, t_ori), total=len(t_ori)):
            if res_tuple is not None:
                l_res_tuple.append(res_tuple)

    flight_df = pd.DataFrame.from_records(l_res_tuple, columns=ResRow._fields)

    flight_df.to_pickle(full_path)

    print(flight_df)
    _duration = pd.Timestamp("now") - _start
    print(f"Loaded  in {_duration} and saved at {full_path}")

    return flight_df


def assign_wtc(flight_df, wtc_df):
    """ "
    Take a dataframe based on flight granularity with already assigned aircraft data

    Input DataFrame has to have 'typecode' and 'icao_type' fields
    """
    return flight_df.merge(
        wtc_df, left_on="typecode", right_on="icao_type", how="left"
    ).drop("icao_type", axis="columns")


def count_wtc(dep_arr_df):
    """
    Count WTC values and one hot encode
    """
    wtc_cols = ["H", "L", "L/M", "M"]
    dep_arr_df.loc[:, wtc_cols] = pd.get_dummies(dep_arr_df.icao_wtc)

    wtc_counts_df = (
        dep_arr_df.sort_values("on_runway_time")
        .set_index("on_runway_time")
        .rolling("30T")[wtc_cols]
        .agg("sum")
    )
    wtc_counts_df = wtc_counts_df.add_suffix(
        "_count"
    )  # dep_arr_df.loc[:, [f"{s}_count" for s in wtc_cols]]
    dep_arr_df = pd.merge_asof(
        dep_arr_df.sort_values("on_runway_time"),
        wtc_counts_df,
        left_on="on_runway_time",
        right_index=True,
    )
    return dep_arr_df


def assign_wtc_ratios(flight_df, wtc_cols=None, nb_movement_col="nb"):
    """
    Suppose that the dataframe given has WTC counts (H_count, M_count...) and Airport congestion columns (nb...)
    Be careful to keep the same time window for these precalculations
    """
    if wtc_cols is None:
        wtc_cols = ["H_count", "L_count", "L/M_count", "M_count"]
    for wtc in wtc_cols:
        flight_df[f"{wtc}_ratio"] = flight_df[wtc] / flight_df[nb_movement_col]
        flight_df.loc[flight_df[f"{wtc}_ratio"] > 1, f"{wtc}_ratio"] = 1
        flight_df.loc[flight_df[f"{wtc}_ratio"] < 0, f"{wtc}_ratio"] = 0
    return flight_df


def runway_configuration_realcounts(df):
    """
    TKOF RWY28/16 LDG RWY14     NORTH0
    TKOF RWY10 LDG RWY14       NORTH1
    TKOF RWY32/34 LDG RWY28     EAST
    TKOF RWY32 LDG RWY34        SOUTH
    """
    df = df.loc[df.runway != "N/A"]
    dep = df.loc[df.mvt_type == "DEP"]
    arr = df.loc[df.mvt_type == "ARR"]
    counts_dep = dep.runway.value_counts()
    counts_arr = arr.runway.value_counts()

    if sum(counts_arr) == 0 and sum(counts_dep) == 0:
        return None
    if sum(counts_arr) == 0:
        rwy_dep_name, rwy_dep_nb = (
            counts_dep.idxmax(),
            counts_dep.loc[counts_dep.idxmax()],
        )
        if rwy_dep_name == "28" or rwy_dep_name == "16":
            return pd.Series({"config": "EAST", "nb_dep": sum(counts_dep), "nb_arr": 0})
        if rwy_dep_name == "10":
            return pd.Series(
                {
                    "config": "NORTH1",
                    "nb_dep": sum(counts_dep),
                    "nb_arr": 0,
                }
            )
        if rwy_dep_name == "32":
            return pd.Series({"config": "EAST", "nb_dep": sum(counts_dep), "nb_arr": 0})
        return None
    if sum(counts_dep) == 0:
        rwy_arr_name, rwy_arr_nb = (
            counts_arr.idxmax(),
            counts_arr.loc[counts_arr.idxmax()],
        )
        if rwy_arr_name == "34":
            return pd.Series(
                {"config": "SOUTH", "nb_dep": 0, "nb_arr": sum(counts_arr)}
            )
        if rwy_arr_name == "28":
            return pd.Series({"config": "EAST", "nb_dep": 0, "nb_arr": sum(counts_arr)})
        if rwy_arr_name == "14":
            return pd.Series(
                {"config": "NORTH0", "nb_dep": 0, "nb_arr": sum(counts_arr)}
            )

    rwy_dep_name, rwy_dep_nb = (
        counts_dep.idxmax(),
        counts_dep.loc[counts_dep.idxmax()],
    )
    rwy_arr_name, rwy_arr_nb = (
        counts_arr.idxmax(),
        counts_arr.loc[counts_arr.idxmax()],
    )

    if rwy_arr_name == "34":  # TKOF RWY32 LDG RWY34
        return pd.Series(
            {
                "config": "SOUTH",
                "nb_dep": len(dep),
                "nb_arr": len(arr),
            }
        )
    if rwy_arr_name == "28":  # TKOF RWY32/34 LDG RWY28
        return pd.Series({"config": "EAST", "nb_dep": len(dep), "nb_arr": len(arr)})
    if rwy_arr_name == "14":
        if rwy_dep_name == "10":  #  TKOF RWY10 LDG RWY14
            return pd.Series(
                {
                    "config": "NORTH1",
                    "nb_dep": len(dep),
                    "nb_arr": len(arr),
                }
            )
        else:  #  TKOF RWY28/16 LDG RWY14
            return pd.Series(
                {
                    "config": "NORTH0",
                    "nb_dep": len(dep),
                    "nb_arr": len(arr),
                }
            )
    return None


def count_movements_and_runway_conf(flight_df_both, wsize=pd.Timedelta("30T")):
    """
    Add the following columns to the given flight dataframe : ['config', 'nb_arr', 'nb_dep', 'nb']
    Suppose that flight_df_both has "flight_id", "on_runway_time", "runway", "mvt_type" as columns
    mvt_type should take values in ['DEP', 'ARR']
    """
    fdfb = flight_df_both.sort_values("first_movement_start").set_index(
        "first_movement_start"
    )[["flight_id", "on_runway_time", "runway", "mvt_type"]]
    rwy_conf_df = pd.DataFrame()
    for df in fdfb.rolling(wsize):
        row = runway_configuration_realcounts(df)
        if row is not None:
            row["flight_id"] = df.iloc[-1].flight_id
            rwy_conf_df = rwy_conf_df.append(row, ignore_index=True)
    rwy_conf_df["nb"] = rwy_conf_df["nb_arr"] + rwy_conf_df["nb_dep"]
    return flight_df_both.merge(rwy_conf_df, how="left", on="flight_id")


def count_movements(flight_df_both, wsize=pd.Timedelta("30T")):
    """
    Suppose that flight_df_both contains a columns 'mvt_type' which attribute to a flight if its a 'DEP' or an 'ARR'
    """
    wsize = pd.Timedelta("30T")
    fdfb = flight_df_both.sort_values("first_movement_start").set_index(
        "first_movement_start"
    )
    fdfb[["nb_arr", "nb_dep"]] = (
        pd.get_dummies(fdfb["mvt_type"]).rolling(wsize).agg("count")
    )
    fdfb["nb_movement"] = fdfb["nb_arr"] + fdfb["nb_dep"]
    return fdfb.reset_index()


def compute_sifi(flight_df, col_name="sifi_DEP", mvt_type="DEP"):
    """

    Based on end_pb and first_movement_start columns to know how many aircraft are taxiing while one is during pushback
    Consider passing only departures in flight_df (dep_df) counts should be specific
    to that type of movement

    Equivalent to SIFI by Yin et al. 2018
    denoting the number of taxiing departures and arrivals,
    respectively, when d0 is being pushed back from the gate.

    Equivalent to NDepDep in Wang et al. 2021
    """
    for i, r in flight_df.sort_values("first_movement_start").iterrows():
        if (
            r.end_pb is not None
            and r.first_movement_start is not None
            and r.end_pb == r.end_pb
            and r.first_movement_start == r.first_movement_start
        ):
            t0 = r.first_movement_start
            t1 = r.end_pb
            assert t0 is not None and t1 is not None
            # count overlaps between other flights duration and current flight pushback
            temp = flight_df.loc[
                (flight_df.index != i)
                & (flight_df.mvt_type == mvt_type)
                & ~(
                    (flight_df.first_movement_start > t1)
                    | (flight_df.on_runway_time < t0)
                )
            ]
            flight_df.loc[i, col_name] = len(temp)
    return flight_df


def compute_aqli(dep_arr_df, mvt_type="DEP"):
    """
    Based on end_pb and first_movement_start columns to know how many aircraft are taxiing while one is during pushback
    Consider passing only departures in flight_df (dep_df) counts should be specific
    to that type of movement

    refers to the number of departures and arrivals, given the dataframe given, whose taxiing period has
    overlap with the taxiing period of d0
    """
    flight_df = dep_arr_df  # to not change directly the input dataframe
    for i, r in flight_df.sort_values("first_movement_start").iterrows():
        if r.end_pb == r.end_pb and r.first_movement_start == r.first_movement_start:
            t0 = r.first_movement_start
            t1 = r.on_runway_time
            # count overlaps between other flights duration and current flight pushback
            temp = flight_df.loc[
                (flight_df.index != i)
                & (flight_df.mvt_type == mvt_type)
                & (flight_df.on_runway_time > t0)
                & (flight_df.on_runway_time < t1)
            ]
            flight_df.loc[i, f"aqli_{mvt_type}"] = len(temp)
    flight_df[f"aqli_{mvt_type}"] = flight_df[f"aqli_{mvt_type}"].fillna(0)
    return flight_df


def compute_scfi(dep_arr_df, mvt_type="DEP"):
    """
    Based on end_pb and first_movement_start columns to know how many aircraft are taxiing while one is during pushback
    Consider passing only departures in flight_df (dep_df) counts should be specific
    to that type of movement

    refers to the number of
    departures and arrivals, given the dataframe given, whose taxiing period has
    overlap with the taxiing period of d0 . The
    """
    flight_df = dep_arr_df  # to not change directly the input dataframe
    for i, r in flight_df.sort_values("first_movement_start").iterrows():
        if r.end_pb == r.end_pb and r.first_movement_start == r.first_movement_start:
            t0 = r.first_movement_start
            t1 = r.on_runway_time
            # count overlaps between other flights duration and current flight pushback
            temp = flight_df.loc[
                (flight_df.index != i)
                & (flight_df.mvt_type == mvt_type)
                & ~(
                    (flight_df.first_movement_start > t1)
                    | (flight_df.on_runway_time < t0)
                )
            ]
            flight_df.loc[i, f"scfi_{mvt_type}"] = len(temp)
    flight_df[f"scfi_{mvt_type}"] = flight_df[f"scfi_{mvt_type}"].fillna(0)
    return flight_df


def compute_avgspeeedlastXaircrafts(flight_df, n=10):

    for i, r in flight_df.sort_values("on_runway_time").iterrows():
        takeoff_before_df = flight_df.loc[
            flight_df.on_runway_time < r.first_movement_start
        ]
        avg_speed = (
            takeoff_before_df.sort_values("on_runway_time").iloc[-n:].avg_speed.mean()
        )
        flight_df.loc[i, f"AvgSpeedLast{n}ac"] = avg_speed
    return flight_df


def compute_time_since_last_departure(flight_df):
    """
    Takes a flight dataframe with a first_movement_start attribute and compute the duration between each start of movement
    """
    time_between_dep = (
        flight_df.query("parking_position==parking_position")
        .sort_values("first_movement_start")
        .first_movement_start.diff()
        .dt.total_seconds()
        / 60
    )  # .hist(bins=20, range=(0,30))
    return pd.concat([flight_df, time_between_dep], axis=1)


def compute_avg_duration(
    dep_df,
    dt="30T",
    time_col="on_runway_time",
    on_col="taxi_holding_time_minutes",
    col_name="avg_delay",
    internal_query=None,
    tolerance=pd.Timedelta("2T"),
):
    # Index is on_runway_time because it makes more sense to compute average
    # ground delays with flights having completed their taxi
    df = dep_df
    if internal_query is not None:
        df = dep_df.query(internal_query)

    avg_delays = (
        df.set_index(time_col)
        .sort_index()[on_col]
        .rolling(dt, closed="left")  # can be a feature to help predict next delay
        .agg("mean")
        .rename(col_name)
    )
    dep_df = pd.merge_asof(
        dep_df.sort_values(time_col),
        avg_delays,
        left_on=time_col,
        right_index=True,
        tolerance=tolerance,
    )
    return dep_df


def onehot_col(flight_df, on_col, drop_col=False):
    """ Takes the flights dataframe and replace the runway col by n col rwy_X, with X each runway """
    res = pd.concat(
        [flight_df, pd.get_dummies(flight_df[on_col], prefix=on_col)], axis=1
    )
    if drop_col:
        return res.drop(on_col, axis=1)
    else:
        return res


def assign_asma_mode(asma_df):
    # TODO move to external_data_.py
    modes = pd.DataFrame.from_records(
        [
            {"initial_flow": "24-72", "cluster": 0, "mode": "nominal"},
            {"initial_flow": "24-72", "cluster": 1, "mode": "busy"},
            {"initial_flow": "24-72", "cluster": 2, "mode": "disrupted"},
            {"initial_flow": "24-72", "cluster": 3, "mode": "disrupted"},
            {"initial_flow": "90-132", "cluster": 0, "mode": "nominal"},
            {"initial_flow": "90-132", "cluster": 1, "mode": "busy"},
            {"initial_flow": "90-132", "cluster": 2, "mode": "disrupted"},
            {"initial_flow": "162-216", "cluster": 0, "mode": "nominal"},
            {"initial_flow": "162-216", "cluster": 1, "mode": "busy"},
            {"initial_flow": "162-216", "cluster": 2, "mode": "disrupted"},
            {"initial_flow": "240-276", "cluster": 0, "mode": "nominal"},
            {"initial_flow": "240-276", "cluster": 1, "mode": "disrupted"},
            {"initial_flow": "312-354", "cluster": 0, "mode": "nominal"},
            {"initial_flow": "312-354", "cluster": 1, "mode": "nominal"},
            {"initial_flow": "312-354", "cluster": 2, "mode": "busy"},
            {"initial_flow": "312-354", "cluster": 3, "mode": "disrupted"},
        ]
    )
    return asma_df.merge(modes, on=["initial_flow", "cluster"], how="left")


def twy_times2occ(df_times):
    """Given a dataframe with entry and exit times, convert it to get the occupancy over time"""
    new_df = df_times.melt(var_name="status", value_name="time").sort_values("time")
    new_df["counter"] = new_df["status"].map({"start": 1, "stop": -1}).cumsum()
    new_df.loc[new_df.counter < 0, "counter"] = 0
    return new_df


def rolling_max(df, dt_str="10T", time_col_name="time"):
    """Convert a occupancy dataframe (with counts, not entry and exit times) to a 'max occupancy'
    dataframe where"""
    return (
        df.sort_values(time_col_name)
        .set_index(time_col_name)
        .rolling(dt_str)
        .max()
        .reset_index()
    )


def assign_twy_occ(dep_arr_df, d_twy, col_suffix="_occ"):
    """
    Affect number of aircraft present at the same time for each taxiway
    Adds columns : all keys of d_twy with '_occ' as a suffixe
    """
    for i, df in d_twy.items():
        new_df = twy_times2occ(df)
        dep_arr_df = (
            pd.merge_asof(
                dep_arr_df.sort_values("first_movement_start"),
                new_df,
                left_on="first_movement_start",
                right_on="time",
                direction="backward",  # In order to not preshot the entry of a future aircraft on a twy
            )
            .drop(["status", "time"], axis=1)
            .rename(columns={"counter": f"{i}{col_suffix}"})
        )
        dep_arr_df[f"{i}_occ"] = dep_arr_df[f"{i}{col_suffix}"].fillna(0)
    return dep_arr_df


def assign_twy_rolling_max(dep_arr_df_twy, twy_cols, dt_str="30T"):
    """ Consider using assign_twy_occ() on the input flight dataframe to assign taxiways occupancy"""
    twy_max_occ = rolling_max(
        dep_arr_df_twy[["first_movement_start"] + twy_cols],
        dt_str,
        time_col_name="first_movement_start",
    )
    dep_arr_df_twy = dep_arr_df_twy.drop(twy_cols, axis=1)
    return pd.merge_asof(
        dep_arr_df_twy.sort_values("first_movement_start"),
        twy_max_occ,
        on="first_movement_start",
    )


#####################################################################

if __name__ == "__main__":
    data_dir = "../data/intermediate/"
    dep_fn = "taxi_zurich_2019_takeoff.pkl"
    arr_fn = "taxi_zurich_2019_landin_v3.pkl"

    _start = pd.Timestamp("now")
    print(f"started at {_start}")

    """traf = (
        Traffic.from_file(data_dir + dep_fn)
        .aircraft_data()
        .cumulative_distance()
        .eval(desc="", max_workers=23)
    )"""
    v = 27
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

    traf = (
        Traffic.from_file(data_dir + arr_fn)
        .aircraft_data()
        .cumulative_distance()
        .eval(desc="", max_workers=20)
    )
    fdf = extract_dep_info(traf, v, fn="flight_df_ARR")
    _duration = pd.Timestamp("now") - _start
    print(f"done in {_duration}")