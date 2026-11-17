# -*- coding: utf-8 -*-
"""
local_config_example.py - copy to local_config.py and fill in your own paths. local_config.py is NEVER uploaded
(build_release.py refuses to package it). Every setting is optional; the scripts fall back to the defaults shown.
"""
# public data (WM-811K LSWMD.pkl, MixedWM38.npz); searched recursively inside this folder
PUBLIC_DATA_DIR = "data"
WM811K_PKL = ""            # explicit path to LSWMD.pkl (empty = search PUBLIC_DATA_DIR)
MIXEDWM38_NPZ = ""         # explicit path to MixedWM38.npz (empty = search PUBLIC_DATA_DIR)

# output folders
OUT_CORE = "results_core_v20"
OUT_EXTENDED = "results_extended_v20"
OUT_REVISION = "results_revision_v20"
CSV_PACKAGE = "csv_package_v20"

# confidential production data (only on the machine that holds them)
PARTNER_DATA_DIR = ""      # folder with the partner exports
PLANTB_LABEL_CSV = ""      # lot labels (lot_id,status)
LOADER_DIR = ""            # folder of production_loaders.py if it is not next to the scripts
# PLANTB_SAME_COATERS = ["...", "..."]   # optional; default: tools with >= 5 normal and >= 1 abnormal lot
# FABA_FILE_GLOB = "..._{kind}.csv"      # optional; file pattern of the Fab A exports
