import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import altair as alt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from collections import namedtuple

import os


ResMetrics = namedtuple(  # create a personalized tuple
    "ResMetrics", "mae mse acc1min acc3min acc5min r2"
)


def within_Xmin(real, pred, m=1):
    assert len(real) == len(pred)
    c = 0
    for r, p in zip(real, pred):
        if abs(r - p) < m:
            c += 1
    return c / len(real) * 100


def plot_results(labels_test, predicted):
    # plot pred vs truth
    data_chart = pd.DataFrame(
        {"true": labels_test.values.reshape((-1,)), "pred": predicted}
    )
    predVStrue_chart = alt.Chart(data_chart).mark_circle(opacity=0.5).encode(
        x="true", y="pred"
    ) + alt.Chart(
        pd.DataFrame({"x": np.linspace(0, 25, 10), "y": np.linspace(0, 25, 10)})
    ).mark_line(
        color="black"
    ).encode(
        x="x", y="y"
    )
    return predVStrue_chart


def plot_feature_importance(
    feature_columns,
    clf,  # saveas="../prediction_results/result.png"
):
    col = feature_columns
    # modelname.feature_importance_
    y = clf.feature_importances_
    # plot
    fig, ax = plt.subplots()
    width = 0.2  # the width of the bars
    ind = np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y, width, color="green")
    ax.set_yticks(ind + width / 10)
    ax.set_yticklabels(col, minor=False)
    plt.title("Feature importance in RandomForest Classifier")
    plt.xlabel("Relative importance")
    plt.ylabel("feature")
    plt.figure(figsize=(5, 7))
    fig.set_size_inches(6.5, 6.5, forward=True)
    return fig


def results(labels_test, predicted):
    mae = mean_absolute_error(labels_test.values.reshape((-1,)), predicted)
    mse = mean_squared_error(labels_test.values.reshape((-1,)), predicted)
    acc1min = within_Xmin(labels_test.values.reshape((-1,)), predicted)
    acc3min = within_Xmin(labels_test.values.reshape((-1,)), predicted, m=3)
    acc5min = within_Xmin(labels_test.values.reshape((-1,)), predicted, m=5)
    r2 = r2_score(labels_test.values.reshape((-1,)), predicted)

    return ResMetrics(
        mae=mae, mse=mse, acc1min=acc1min, acc3min=acc3min, acc5min=acc5min, r2=r2
    )


def train_rf(
    flight_df,
    feature_columns,
    target_column,
    query,
    dirname="../prediction_results/rf/",
):
    """
    function that will train and plot performance metric
    """

    flight_df = flight_df.query(query)

    data_df = flight_df[feature_columns + [target_column]].dropna()

    input_df = data_df[feature_columns]
    target_df = data_df[[target_column]]

    data_train, data_test, labels_train, labels_test = train_test_split(
        input_df,
        target_df,
        test_size=0.20,
    )

    clf = RandomForestRegressor(n_estimators=150, min_samples_split=10)

    # fit the model on the whole dataset
    clf.fit(data_train, labels_train)
    # make a single prediction
    predicted = clf.predict(data_test)

    res_metrics = results(labels_test, predicted)

    res_txt = "Results for Random Forest on the following features :\n"
    res_txt += " ".join(feature_columns) + "\n"
    res_txt += f"Target var : {target_column}\n"
    res_txt += "Selecting the subset with following query : \n" + query + "\n"
    res_txt += f"mae={res_metrics.mae}\nmse={res_metrics.mse}\nacc1min={res_metrics.acc1min}\nacc3min={res_metrics.acc3min}\nacc5min={res_metrics.acc5min}\nAnd R2 = {res_metrics.r2} "

    print(res_txt)

    corr_chart = plot_results(labels_test, predicted)
    feature_chart = plot_feature_importance(feature_columns, clf)

    os.makedirs(os.path.dirname(dirname), exist_ok=True)
    with open(dirname + "res.txt", "w") as text_file:
        text_file.write(res_txt)

    feature_chart.savefig(dirname + "features.png")
    corr_chart.save(dirname + "corr_chart.html")

    return res_metrics


def main():
    dep_arr_df = pd.read_pickle("../data/processed/dep_arr_df6_twypca.pkl")
    twy_cols = [
        "H_occ",
        "K_occ",
        "B_occ",
        "F_occ",
        "L_occ",
        "J_occ",
        "E_occ",
        "D_occ",
        "C_occ",
        "N_occ",
        "A_occ",
        "M_occ",
        "Y_occ",
        "R_occ",
        "G_occ",
        "Z_occ",
        "P_occ",
    ]
    twy_pca_cols = [
        "twy_pca_0",
        "twy_pca_1",
        "twy_pca_2",
    ]
    apt_congest_cols = ["nb_arr", "nb_dep", "nb", "hh_num"]
    sifi_cols = ["aqli_DEP", "scfi_DEP", "aqli_ARR", "scfi_ARR"]
    rwy_cols = ["rwy__10", "rwy__14", "rwy__16", "rwy__28", "rwy__32", "rwy__34"]
    taxi_path_cols = ["taxi_dist", "angle_sum", "turnaround_minutes"]  # To add if relevant
    asma_cols = [
        "nominal",
        "busy",
        "disrupted",
        "not_nom_arrival_ratio",
    ]
    weather_cols = [
        "press",
        "temp",
        "atmap_visibility",
        "atmap_wind",
        "atmap_precipitation",
        "atmap_freezing",
        "atmap_danger",
    ]
    wtc_counts_cols = [
        "H_count",
        "L_count",
        "L/M_count",
        "M_count",
    ]
    wtc_ratios_cols = [
        "H_count_ratio",
        "L_count_ratio",
        "L/M_count_ratio",
        "M_count_ratio",
    ]
    
    wtc_one_hot_col = ["H", "L", "L/M", "M"]

    all_columns = (
        twy_cols
        + apt_congest_cols
        + sifi_cols
        + rwy_cols
        + taxi_path_cols
        + asma_cols
        + weather_cols
        + wtc_counts_cols
        + wtc_ratios_cols
        + twy_pca_cols
        + wtc_one_hot_col
    )

    base_dir = "../prediction_results/"
    targets = [
        "real_dur_minutes",
        "total_holding_time_minutes",
        "avg_delay_30min",
        "avg_delay_60min",
    ]


    test_name='01_basic'
    train_rf(
        dep_arr_df,
        all_columns,
        "real_dur_minutes",
        query="config=='NORTH0' & end_pb==end_pb & mvt_type=='DEP'",
        dirname=base_dir + test_name,
    )

    scenarios = [
        sifi_cols,
        # all_columns,
        # rwy_cols + taxi_path_cols + asma_cols + weather_cols,
    ]

    # l_res = []
    # l_params = []

    # for target_col in targets:
    #     for i, features in enumerate(scenarios):
    #         test_name = f"{i}" + target_col + "/" + "_".join(features)

    #         res_metrics = train_rf(
    #             dep_arr_df,
    #             features,
    #             target_col,
    #             query="config=='NORTH0' & end_pb==end_pb & mvt_type=='DEP'",
    #             dirname=base_dir + test_name,
    #         )
    #         l_res.append(res_metrics)
    #         l_params.append(test_name)
    # pd.Series({})


if __name__ == "__main__":
    main()
