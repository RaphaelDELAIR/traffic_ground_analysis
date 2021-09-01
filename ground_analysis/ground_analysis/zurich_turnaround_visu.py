# noqa: E402

# %%
import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()

turnaround = pd.read_pickle("turnaround/zurich_turnaround.pkl")
# %%
alt.Chart(
    turnaround.query('turnaround < "5H"').assign(
        turnaround=lambda df: df.turnaround.dt.total_seconds() / 60
    )
).mark_bar().encode(
    alt.Y("count()"), alt.X("turnaround", bin=alt.Bin(maxbins=30))
).transform_filter(
    "utchours(datum.AOBT) < 10"
)

# %%
base = (
    alt.Chart(
        turnaround.query('turnaround < "6H"').assign(
            turnaround=lambda df: df.turnaround.dt.total_seconds() / 60
        )
    )
    .transform_filter(
        "(('2019-10-28' >= datum.start) & (datum.start >= '2019-10-21')) |"
        "(('2019-11-18' >= datum.start) & (datum.start >= '2019-11-04'))"
    )
    .mark_circle()
    .encode(
        # alt.Y(
        #    "median(turnaround)",
        #    title=None,  # "Median of stand duration",
        #    scale=alt.Scale(
        #        # type="log", domain=(10, 1000),
        #        scheme="yelloworangered"
        #    ),
        #    axis=None,
        # ),
        # alt.Color("count()", title="Number of aircraft"),
        alt.Y(
            "utcmonthdate(AOBT):O",
            title=None,
            # header=alt.Header(labelAngle=0, labelAlign="right", format="%b %d"),
        ),
        alt.X("utchours(AOBT)", title="AOBT"),
        alt.Fill(
            "median(turnaround)",
            title="Stand duration (minutes)",
            scale=alt.Scale(type="log", domain=(50, 300), scheme="yelloworangered"),
        ),
        alt.Size("count()", title="Number of aircraft"),
    )
    .properties(
        width=500,
    )  # height=40, bounds="flush")
    .configure_view(stroke=None)
    .configure_facet(spacing=0)
    .configure_title(font="Fira Sans", fontSize=18, anchor="start")
    .configure_axis(
        labelFont="Fira Sans",
        labelFontSize=13,
        titleFont="Ubuntu",
        titleFontSize=16,
    )
    .configure_legend(
        orient="bottom",
        labelFont="Ubuntu",
        labelFontSize=14,
        titleFont="Ubuntu",
        titleFontSize=14,
        padding=10,
    )
    .configure_header(
        labelFont="Fira Sans",
        labelFontSize=15,
        labelAngle=0,
        labelAlign="left",
        format="%b %d",
    )
)
base
# %%
base.save("zurich_stand_duration.svg")
# %%
