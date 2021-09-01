from .lrfinder import LRFinder
import pandas as pd

dep_arr_df = pd.read_pickle("../data/processed/dep_arr_df3.pkl")

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

flight_df = dep_arr_df.query("avg_outbound_delay_lastXmin==avg_outbound_delay_lastXmin")
