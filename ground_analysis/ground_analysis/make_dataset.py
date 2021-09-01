import concurrent.futures
import multiprocessing
import os

import altair as alt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from metar import Metar
from pandas.errors import ParserError
from traffic.core import Traffic
from traffic.data import airports

from itertools import repeat

from .utils import daterange

"""
Objective is to take raw OPEN SKY ADS B data and take only trajectories in the vicinity of zurich  with landingz

Also used to load external data (weather online)

Results are stored in data/intermediate

Note: Departures are not extracted from raw data for now as the ile was 
already generated
"""


################# Departures #################
def load_ground_dep_data_parallelized(
    start,
    stop,
    min_nb_data=30,
    airport="LSZH",
    alt_max=3000,
    data_dir="../data/",
    traffic_file="taxi_zurich_2019_takeoff.pkl",
):
    """
    Returns ground traffic between start and stop timestamps
    Load raw ADS B Data and returns departures until they reach 5nm

    use  airport_vicinity() in flight_processing
    use __process_flight in ground_functions

    """
    raise NotImplemented


################ Landings ###################


def extract_landings_ground_traffic(
    day, landingzurich2019_day_df, rawtraf_dir="../data/raw/LSZH_history/"
):
    """
    day is the day to be analysed, landingzurich2019_day_df is a dataframe containing callsign, icao24 and flight_id of all
    flights of the corresponding day from the dataset landing_zurich2019

    consider using :
    >>> landingzurich2019_day_df = t_landingzurich2019_day.summary(['callsign', 'icao24', 'flight_id', 'start', 'stop', 'origin'])

    """
    day_string = day.strftime("%Y-%m-%d")
    fn_rawtraf = f"LSZH_{day_string}_history.pkl.gz"
    t_total = (
        Traffic.from_file(rawtraf_dir + fn_rawtraf)
        .assign_id()
        .eval(desc="", max_workers=23)
    )
    t_total_nearapt = (
        t_total.distance(airports["LSZH"])
        .query("distance<5")
        .eval(desc="", max_workers=23)
        .summary(["flight_id", "start", "callsign", "icao24"])
    )

    l_flights_assigned_ids = []

    # iterate over landingzurich instead of the raw traffic
    for i, r in landingzurich2019_day_df.iterrows():
        # we then try to seek for the corresponding
        t_sameflight = t_total_nearapt.query(
            f"callsign=='{r.callsign}' & icao24=='{r.icao24}'"
        )
        if t_sameflight is not None:
            if (
                len(t_sameflight) == 1
            ):  # Exactly one flight matching => we return it with corresponding id
                matching_fid = t_sameflight.iloc[0].flight_id
            else:
                # several flights in raw traffic were found => we should select the right one
                # => the one arriving to the runway at the closest time of the 'stop' field extracted from landing_zurich_2019
                # As a proxy we use the start of entering in the 5nm circle as the on runway time for the flight coming from the raw traffic
                # t_sameflight.loc[:, 'delta_arriving_time']=t_sameflight['start'].apply(lambda x: (r.stop - x))#.dt.total_seconds()
                t_sameflight = t_sameflight.assign(
                    delta_arriving_time=lambda x: (
                        r.stop - x["start"]
                    ).dt.total_seconds()
                )
                id_match = t_sameflight.query(
                    "delta_arriving_time>0"
                ).delta_arriving_time.idxmin()
                matching_fid = t_sameflight.loc[id_match, "flight_id"]
            new_f = (
                t_total[matching_fid]
                .after(
                    t_total_nearapt.query(f"flight_id=='{matching_fid}'").iloc[0].start
                )
                .assign_id(r.flight_id)
            )  # f.after(f_nearby_apt.start).assign_id(fid)
            if r.origin is not None and isinstance(
                r.origin, str
            ):  # to include flights from landing_zurich_2019 having None as origin
                new_f = new_f.query(f'origin=="{r.origin}"')
            l_flights_assigned_ids.append(new_f)
    return Traffic.from_flights(l_flights_assigned_ids)


def load_extract_landing_ground_traffic_para(
    from_date,
    to_date,
    num_processes=None,
):
    """
    returns and save a t_arr pickle file containing all flights arriving to the airport cut in a range of 5nm
    and with corresponding flight_id from the dataset landing_zurich_2019
    """
    from traffic.data.datasets import landing_zurich_2019

    l_days = [single_date for single_date in daterange(from_date, to_date)]
    l_landingzurich2019_day_df = [
        landing_zurich_2019.between(
            single_date, single_date + pd.Timedelta("1d")
        ).summary(["callsign", "icao24", "flight_id", "start", "stop", "origin"])
        for single_date in daterange(from_date, to_date)
    ]

    l_traf = []

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
        for traf in pool.map(
            extract_landings_ground_traffic, l_days, l_landingzurich2019_day_df
        ):
            l_traf.append(traf)

    res_traf = sum(l_traf)

    return res_traf


############## External ##################


def parse_metars(start, stop):
    if os.path.exists(f"../data/metar_df{start}_{stop}.pkl"):
        return pd.read_pickle(f"../data/metar_df{start}_{stop}.pkl")

    metar_df = pd.DataFrame()

    date = start

    while date <= stop:
        d = date.strftime("%Y%m%d")
        page = requests.get(
            f"http://weather.uwyo.edu/cgi-bin/wyowx.fcgi?TYPE=metar&DATE={d}&STATION=LSZH"
        )
        soup = BeautifulSoup(page.content, "html.parser")
        metars = (
            list(list(list(soup.children)[0].children)[3].children)[3]
            .get_text()
            .split("\n")[1:-1]
        )
        for m in metars:
            try:
                obs = Metar.Metar(m, month=date.month, year=date.year)
            except ParserError as e:
                print(e)
                continue
            except Exception as e:
                print(e)
                continue
            metar_df = metar_df.append(
                {
                    "date": obs.time,
                    "temp": obs.temp.value() if obs.temp is not None else None,
                    "press": obs.press.value() if obs.temp is not None else None,
                    "wind_dir_name": obs.wind_dir.compass()
                    if obs.wind_dir is not None
                    else None,
                    "wind_dir": obs.wind_dir.value()
                    if obs.wind_dir is not None
                    else None,
                    "wind_dir_peak": obs.wind_dir_peak.value()
                    if obs.wind_dir_peak is not None
                    else None,
                    "wind_speed": obs.wind_speed.value()
                    if obs.wind_speed is not None
                    else None,
                    "wind_speed_peak": obs.wind_speed_peak.value()
                    if obs.wind_speed_peak is not None
                    else None,
                    "wind_gust": obs.wind_gust.value()
                    if obs.wind_gust is not None
                    else None,
                    "vis": obs.vis.value() if obs.vis is not None else None,
                    "max_vis": obs.max_vis.value() if obs.max_vis is not None else None,
                    "runway": obs.runway,  # weather,
                    "precip_1hr": obs.precip_1hr.value()
                    if obs.precip_1hr is not None
                    else None,
                    "press_sea_level": obs.press_sea_level.value()
                    if obs.press_sea_level is not None
                    else None,
                    "snowdepth": obs.snowdepth.value()
                    if obs.snowdepth is not None
                    else None,
                },
                ignore_index=True,
            )
        date += pd.Timedelta(days=1)
    metar_df["date"] = metar_df["date"].dt.tz_localize("utc")
    metar_df.to_pickle(f"../data/metar_df{start}_{stop}.pkl")
    return metar_df


def load_atmap(filename="../data/external/lszh_metar_atmap.csv"):
    atmap_df = (
        pd.read_csv(filename, parse_dates=["time_utc"])
        .drop("id", axis="columns")
        .assign(
            atmap_score=lambda r: (
                r.atmap_visibility
                + r.atmap_wind
                + r.atmap_precipitation
                + r.atmap_freezing
                + r.atmap_danger
            )
            / 5
        )
    )
    atmap_df["time_utc"] = pd.to_datetime(atmap_df["time_utc"], utc=True)
    atmap_df = atmap_df.assign(hh=lambda r: r["time_utc"].dt.floor("30T")).assign(
        hh_num=lambda r: r.hh.dt.hour + r.hh.dt.minute / 60
    )
    return atmap_df


def load_wtc(fn):
    wtc_df = pd.read_csv(fn, sep=";", usecols=["ICAO Type Designator", "ICAO WTC"])
    wtc_df = wtc_df.dropna().rename(
        columns={"ICAO Type Designator": "icao_type", "ICAO WTC": "icao_wtc"}
    )
    return wtc_df


#####################################################################

if __name__ == "__main__":
    from_date = pd.Timestamp("2019-10-01 00:00:00+00:00")
    to_date = pd.Timestamp("2019-12-01 00:00:00+00:00")

    _start_time = pd.Timestamp("now")
    print(f"Starting at {_start_time}")

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
