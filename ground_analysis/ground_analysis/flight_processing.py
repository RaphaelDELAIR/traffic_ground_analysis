from traffic.core import Traffic, Flight
import traffic.core.geodesy as geo
from traffic.data import airports
from traffic.core.geodesy import distance
from traffic.algorithms.douglas_peucker import douglas_peucker

from pyproj.exceptions import CRSError
from shapely.geometry import Point, LineString, MultiLineString

import concurrent.futures
import multiprocessing

import pandas as pd
from itertools import repeat
import numpy as np

from .airport_processing import get_long_twy


def clean_ground_traj(f):
    cur_sorted = f.sort_values("timestamp", ascending=True)
    coords = cur_sorted.data[["timestamp", "latitude", "longitude"]]

    delta = pd.concat(
        [coords, coords.add_suffix("_1").diff(), coords.add_suffix("_2").diff(2)],
        axis=1,
    )

    delta_1 = delta.iloc[2:]
    d_prec_curr = geo.distance(
        delta_1.latitude.values,
        delta_1.longitude.values,
        (delta_1.latitude + delta_1.latitude_1).values,
        (delta_1.longitude + delta_1.longitude_1).values,
    )
    d_curr_next = geo.distance(
        (delta_1.latitude + delta_1.latitude_1).values,
        (delta_1.longitude + delta_1.longitude_1).values,
        (delta_1.latitude + delta_1.latitude_2).values,
        (delta_1.longitude + delta_1.longitude_2).values,
    )
    d_prec_next = geo.distance(
        delta_1.latitude.values,
        delta_1.longitude.values,
        (delta_1.latitude + delta_1.latitude_2).values,
        (delta_1.longitude + delta_1.longitude_2).values,
    )

    delta_1.loc[:, "prec_curr"] = d_prec_curr
    delta_1.loc[:, "curr_next"] = d_curr_next
    delta_1.loc[:, "prec_next"] = d_prec_next

    delta_1.loc[
        (delta_1.prec_curr > 100) & (delta_1.curr_next > 100), "long_movement"
    ] = True
    delta_1["long_movement"] = delta_1["long_movement"].fillna(False)

    delta_1["to_rm"] = False
    delta_1.loc[
        (delta_1.long_movement)
        & (
            (delta_1.prec_curr > 5 * delta_1.prec_next)
            | (delta_1.curr_next > 5 * delta_1.prec_next)
        ),
        "to_rm",
    ] = True

    delta_1["gs_prec_next"] = (
        delta_1["prec_next"] / delta_1.timestamp_2.dt.total_seconds()
    )

    bearing_prec_curr = geo.bearing(
        delta_1.latitude.values,
        delta_1.longitude.values,
        (delta_1.latitude + delta_1.latitude_1).values,
        (delta_1.longitude + delta_1.longitude_1).values,
    )
    bearing_curr_next = geo.bearing(
        delta_1.latitude.values,
        delta_1.longitude.values,
        (delta_1.latitude + delta_1.latitude_2).values,
        (delta_1.longitude + delta_1.longitude_2).values,
    )
    delta_1["bearing_prec_curr"] = bearing_prec_curr
    delta_1["bearing_curr_next"] = bearing_curr_next

    delta_1["bearing_prec_curr"] = np.where(
        delta_1["bearing_prec_curr"] >= 0,
        delta_1["bearing_prec_curr"],
        360 + delta_1["bearing_prec_curr"],
    )
    delta_1["bearing_curr_next"] = np.where(
        delta_1["bearing_curr_next"] >= 0,
        delta_1["bearing_curr_next"],
        360 + delta_1["bearing_curr_next"],
    )

    delta_1["angle"] = (
        delta_1["bearing_prec_curr"] + (180 - delta_1["bearing_curr_next"])
    ) % 180

    delta_1.loc[(delta_1.gs_prec_next > 50) & (delta_1.angle > 60), "to_rm"] = True
    delta_1.loc[(delta_1.gs_prec_next > 30) & (delta_1.angle > 90), "to_rm"] = True
    delta_1.loc[(delta_1.gs_prec_next > 20) & (delta_1.angle > 120), "to_rm"] = True
    delta_1.loc[(delta_1.gs_prec_next > 16.7) & (delta_1.angle > 130), "to_rm"] = True
    delta_1.loc[(delta_1.gs_prec_next > 13.3) & (delta_1.angle > 140), "to_rm"] = True
    delta_1.loc[(delta_1.gs_prec_next > 10) & (delta_1.angle > 150), "to_rm"] = True

    return delta_1


def airport_vicinity(f):
    f_nearby_apt = f.distance(airports["LSZH"]).query("distance<5")
    if f_nearby_apt is None:
        print(f"{f.callsign} / {f.icao24} is not in the 5nm range")
        return None
    else:
        return f.after(f_nearby_apt.start)


def angle_sum(f, tolerance=15):
    f = f.onground()
    if f is None or len(f) < 3:
        return None
    try:
        f_mask = douglas_peucker(
            df=f.data, tolerance=tolerance, lat="latitude", lon="longitude"
        )
    except RecursionError:
        print(f"Problem for douglas_peucker in angle sum for {f.flight_id}")
        return None
    f_simplified = Flight(f.data.loc[f_mask])
    return (
        f_simplified.cumulative_distance()
        .unwrap("compute_track")
        .diff("compute_track_unwrapped")
        .data.compute_track_unwrapped_diff.abs()
        .sum()
    )


def cumdist_ground(f):
    f = f.onground()
    if f is None:
        return None
    try:
        f_mask = douglas_peucker(
            df=f.data, tolerance=15, lat="latitude", lon="longitude"
        )
    except RecursionError:
        print(f"Problem for douglas_peucker in angle sum for {f.flight_id}")
        return None
    f_data = f.data.loc[f_mask]
    f_simplified = Flight(f_data)
    coords = (
        f_simplified.sort_values("timestamp")
        .data[["timestamp", "latitude", "longitude"]]
        .dropna()
    )
    if len(coords) < 2:
        return None
    df = pd.DataFrame()
    for i in range(len(coords) - 1):
        t = coords.iloc[i + 1].timestamp
        lat1, lon1 = coords.iloc[i].latitude, coords.iloc[i].longitude
        lat2, lon2 = coords.iloc[i + 1].latitude, coords.iloc[i + 1].longitude
        d = geo.distance(lat1, lon1, lat2, lon2)
        df = df.append({"timestamp": t, "dist": d}, ignore_index=True)
    return df.dist.sum()


def ground_movement_type(f, airport):
    if f.destination == airport and f.origin == airport:
        return "BOTH"
    if f.origin == airport:
        return "DEP"
    if f.destination == airport:
        return "ARR"
    else:
        return None


####### Taxiways ##########


def assign_twy(f, airport, long_twy_df=None):
    """
    Assign in a new column 'twy' the name of the long taxiway the point is on

    long_twy_df is a dataframe containing shaely geometry column with corresponding twy
    """
    if f is None:
        return f
    if not (f.origin == airport or f.destination == airport):
        print(
            f"{f.flight_id} has a weird origin ({f.origin}) or destination ({f.destination})"
        )
        return f

    if long_twy_df is None:
        long_twy_df = get_long_twy(airport)
    f_ground = f.onground()
    if f_ground is None:
        return f
    f_ground = f_ground.moving()
    if f_ground is None:
        return f
    pb = f.pushback("LSZH")

    mvt_type = ground_movement_type(f, airport)
    if mvt_type == "BOTH":
        print(f.flight_id, " is both")
        # raise NotImplementedError  # TODO
        return f
    elif mvt_type == "DEP":
        if pb is not None:
            f_ground = f_ground.after(pb.stop)
        if f_ground is None:
            return f
        if f.aligned_on_runway("LSZH").has():
            f_ground = f_ground.before(f.aligned_on_runway("LSZH").max().start)
        else:
            return f
    elif mvt_type == "ARR":
        pp = f.on_parking_position(airport).next()
        if pp is not None:
            f_ground = f_ground.before(pp.start)
        if f_ground is None:
            return f
        if f.aligned_on_runway("LSZH").has():
            f_ground = f_ground.after(f.aligned_on_runway("LSZH").max().stop)
        else:
            return f

    if f_ground is None:
        return f

    try:
        mask = douglas_peucker(
            df=f_ground.data, tolerance=15, lat="latitude", lon="longitude"
        )
    except CRSError as e:
        return f

    simplified_df = f_ground.data.loc[mask]
    df = f.data

    for i in range(1, len(simplified_df)):
        t1 = simplified_df.iloc[i - 1].timestamp
        t2 = simplified_df.iloc[i].timestamp
        p1 = Point(
            simplified_df.iloc[i - 1].longitude, simplified_df.iloc[i - 1].latitude
        )
        p2 = Point(simplified_df.iloc[i].longitude, simplified_df.iloc[i].latitude)

        def extremities_dist(twy):
            p1_proj = twy.interpolate(twy.project(p1))
            p2_proj = twy.interpolate(twy.project(p2))
            d1 = distance(p1_proj.y, p1_proj.x, p1.y, p1.x)
            d2 = distance(p2_proj.y, p2_proj.x, p2.y, p2.x)
            return d1 + d2

        temp_ = long_twy_df.assign(dist=np.vectorize(extremities_dist))
        nearest_twy = temp_["dist"].idxmin()
        f.data.loc[(t1 < df.timestamp) & (f.data.timestamp <= t2), "twy"] = nearest_twy
    return f


def assign_twy_traf(traf, airport, long_twy_df):
    return Traffic.from_flights([assign_twy(f, airport, long_twy_df) for f in traf])


def assign_twy_traf_para(traf, airport, long_twy_df, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    lf = []
    # l_input = [(f, long_twy_df) for f in traf]
    with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
        for f in pool.map(assign_twy, traf, repeat(airport), repeat(long_twy_df)):
            lf.append(f)
    return Traffic.from_flights(lf)


def extract_max_twy_occupancy(traf):
    """
    Take a traffic where twys has been assigned to each flight and
    return the maximum number of aircraft present at the same time for each twy
    of the airport.
    """
    df = pd.DataFrame()
    for f in traf:
        entry_exit_twy = f.data.groupby("twy").agg({"timestamp": ["min", "max"]})
        entry_exit_twy.columns = entry_exit_twy.columns.map("_".join)
        df = df.append(entry_exit_twy)
    df = df.reset_index()
    l_d = []
    for twy_name in df.twy.unique():
        df_twy_name = df.query(f"twy=='{twy_name}'")[["timestamp_min", "timestamp_max"]]
        new_df = df_twy_name.melt(var_name="status", value_name="time").sort_values(
            "time"
        )
        new_df["counter"] = (
            new_df["status"].map({"timestamp_min": 1, "timestamp_max": -1}).cumsum()
        )
        res = new_df["counter"].max()
        l_d.append({"twy": twy_name, "occupancy_max": res})
    if len(l_d) == 0:
        return None
    res = pd.DataFrame(l_d)
    res = res.set_index("twy").transpose()
    return res


def load_twy_occ_hh(traf, t_range="30T"):
    """
    Split traffic in t_range time intervals and extract maximum of aircraft present
    at the same time during each of the time interval t_range
    """
    t_range_ = pd.Timedelta(t_range)
    full_traf = traf
    start = full_traf.start_time.floor(t_range)
    stop = full_traf.end_time
    n_iter = int((stop - start) / t_range_)
    twy_occ = pd.DataFrame()
    for i in range(n_iter + 1):
        start_ = start + (i * t_range_)
        stop_ = start + ((i + 1) * t_range_)
        sample_traf = full_traf.between(start_, stop_)
        if sample_traf is not None:
            _df = extract_max_twy_occupancy(sample_traf)
            if _df is not None and len(_df) >= 1:
                _df.index = [start_]
                twy_occ = twy_occ.append(_df)
    twy_occ = twy_occ.fillna(0)
    return twy_occ
