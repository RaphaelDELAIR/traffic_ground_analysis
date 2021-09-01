import pandas as pd
from traffic.core import Traffic, Flight
from typing import Optional, Iterator


def after_landing(flight) -> Optional[Flight]:
    f = None
    for f in flight.aligned_on_ils("LSZH"):
        continue
    if f is None:
        return flight.last("1T")
    flight = flight.after(f.start)
    f = flight.on_runway("LSZH")
    if f is None:
        return flight.last("1T")
    return flight.after(f.start)


def first_if_exist(x: pd.DataFrame) -> dict:
    if x.shape[0] == 0:
        return {}
    return x.iloc[0].to_dict()


def process(
    inbound: pd.DataFrame, outbound: pd.DataFrame
) -> Iterator[pd.DataFrame]:
    for aircraft, df in inbound.rename(
        columns=dict(start="ALDT", stop="AIBT", callsign="callsign_in")
    ).groupby("icao24"):

        out = outbound.rename(
            columns=dict(first_movement="AOBT", callsign="callsign_out")
        ).query(f'icao24 == "{aircraft}"')

        yield pd.DataFrame.from_records(
            list(
                {
                    **in_line.to_dict(),
                    **first_if_exist(out.query("start > @in_line.AIBT")),
                }
                for _, in_line in df.iterrows()
            )
        )


def main():

    df = pd.read_pickle("zurich_departure_stats.pkl")
    summary = pd.read_pickle("zurich_departure_flight_id.pkl")

    outbound = df.merge(summary)[
        ["callsign", "icao24", "start", "firstseen_min", "first_movement"]
    ].sort_values("start")

    arr_ground = (
        Traffic.from_file("zurich_2019_landing_taxi.pkl")
        .query('origin!="LSZH"')  # type: ignore
        .has("aligned_on_LSZH")
        .cumulative_distance()
        .filter(compute_gs=(17, 53), altitude=(17, 53))
        .query("compute_gs > 10")
        .pipe(after_landing)
        .eval(desc="", max_workers=6)
    )

    inbound = arr_ground.summary(
        ["callsign", "icao24", "start", "stop"]
    ).sort_values("start")

    turnaround = (
        pd.concat(list(process(inbound, outbound)))
        .assign(AOBT=lambda df: pd.to_datetime(df.AOBT, utc=True))
        .query("start==start and AOBT == AOBT")
        .sort_values("AOBT")
        .assign(turnaround=lambda df: df.AOBT - df.AIBT)
    )

    turnaround.to_pickle("zurich_turnaround.pkl")


if __name__ == "__main__":
    main()
