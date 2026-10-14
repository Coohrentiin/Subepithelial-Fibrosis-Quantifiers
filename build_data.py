"""
Build the anonymised, self-contained data assets for the paper repository.

This script is run ONCE by the authors, with access to the original raw OCT
images, to bake the SNR values into the quantifier tables and to replace every
patient-identifying ``Origin`` string with an anonymous code. Its outputs (the
pickle files and CSV files under ``data/``) are the only data the public
analysis pipeline (``main.py``) needs, so no patient image or name ever ships
with the repository.

The anonymisation preserves only the non-identifying structure required to
reproduce the figures:
  * a stable anonymous patient code (e.g. ``P001``),
  * the eye side (OD / OS) where it is encoded in the file name,
  * the post-operative month for the longitudinal PRK/haze analysis,
  * the acquisition index for the repeatability analysis.

It is included for provenance/transparency only. Reviewers do NOT need to run
it: the anonymised assets are already committed under ``data/``.
"""

import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Original (private) project root, used only when this script is re-run by the
# authors to regenerate the anonymised assets from the raw quantifier pickles.
SRC_PKL_DIR = os.path.join(HERE, "..", "Dataset_2025arch")
RAW_DATA_DIR = os.path.join(HERE, "..", "Data")
RAW_TTM_DIR = os.path.join(HERE, "..", "Dataset_2025arch")

OUT_DIR = os.path.join(HERE, "data")

# Column rename used throughout the paper (matches RESULTS_2025.ipynb).
RENAME = {
    "mean": "Relative Depth",
    "IntensityPeak": "Peak Intensity",
    "PeakWidth": "Peak Width",
    "DataCov": "COV",
    "AreaRatio": "Area Ratio",
}


def _snr(origin, folder, appended="_flat-post-ttm"):
    """Signal-to-noise ratio for one acquisition, from the raw + flattened image."""
    import sys
    from PIL import Image
    sys.path.append(os.path.join(HERE, "src"))
    from image_OCT import image_OCT

    im = image_OCT(os.path.join(RAW_DATA_DIR, folder), origin + ".jpg")
    ttm = np.array(Image.open(os.path.join(RAW_TTM_DIR, folder, origin + appended + ".tiff")))
    im.set_OCT_flat(ttm)
    return im.ImageSNR()


def _patient_name(origin):
    """
    Extract the identifying surname token from an Origin string.

    Most Origins start with the surname (``AAAAAA_OS_...``). A few of the small
    internal-healthy files are prefixed by a numeric acquisition id
    (``80553_BBBBBB_OS``); there the surname is the second token. Purely
    anonymous illustration names (``haze 6``) have no surname to hide.
    """
    stem = origin.rstrip(".")
    parts = stem.split("_")
    if parts[0].isdigit() and len(parts) > 1:
        return parts[1]
    return parts[0]


def _anon_key(origins):
    """Map each distinct patient surname to a stable code P001, P002, ..."""
    names = [_patient_name(o) for o in origins]
    order = list(dict.fromkeys(names))  # first-seen order, deterministic
    return {name: f"P{ i+1 :03d}" for i, name in enumerate(order)}


def anonymise_origin(origin, key):
    """
    Replace the patient surname in an Origin string by its anonymous code and
    drop any numeric acquisition-id prefix, keeping only the non-identifying
    structure (eye, month, index).
    """
    stem = origin.rstrip(".")
    parts = stem.split("_")
    if parts[0].isdigit() and len(parts) > 1:
        parts = parts[1:]  # drop numeric id prefix (can be identifying)
    parts[0] = key[parts[0]]
    return "_".join(parts)


def build_main():
    """Healthy_2 (166) and Fuchs_2 (313): the classifier training/validation set."""
    dfs = {}
    for pkl, tag in [("data_set_Q3_healthy_2_mov_win_", "healthy_2"),
                     ("data_set_Q3_fuchs_2_mov_win_", "fuchs_2")]:
        df = pd.read_pickle(os.path.join(SRC_PKL_DIR, pkl + ".pkl")).rename(columns=RENAME)
        dfs[tag] = df
    return dfs


def build_with_snr(pkl, folder):
    """Longitudinal / repeatability sets: bake the SNR marker values in."""
    df = pd.read_pickle(os.path.join(SRC_PKL_DIR, pkl + ".pkl")).rename(columns=RENAME)
    df["SNR"] = [_snr(o.rstrip("."), folder) for o in df["Origin"]]
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build one global anonymisation key so a patient present in several sets
    # (e.g. repeatability and main) keeps the same code everywhere.
    all_origins = []
    for pkl in ["data_set_Q3_healthy_2_mov_win_", "data_set_Q3_fuchs_2_mov_win_",
                "data_set_Q3_haze_2_mov_win_", "data_set_Q3_haze_mov_win_",
                "data_set_Q3_healthy_mov_win_",
                "data_set_Q3_rep_fuchs_mov_win_", "data_set_Q3_rep_healthy_mov_win_"]:
        all_origins += list(pd.read_pickle(os.path.join(SRC_PKL_DIR, pkl + ".pkl"))["Origin"])
    key = _anon_key(all_origins)

    # ------------------------------------------------------------------ main sets
    for pkl, out in [("data_set_Q3_healthy_2_mov_win_", "healthy_2"),
                     ("data_set_Q3_fuchs_2_mov_win_", "fuchs_2")]:
        df = pd.read_pickle(os.path.join(SRC_PKL_DIR, pkl + ".pkl")).rename(columns=RENAME)
        df["Origin"] = [anonymise_origin(o, key) for o in df["Origin"]]
        df.to_pickle(os.path.join(OUT_DIR, f"quantifiers_{out}.pkl"))

    # ------------------------------------------------------------ longitudinal / haze
    df = build_with_snr("data_set_Q3_haze_2_mov_win_", "HAZE_2")
    df["Origin"] = [anonymise_origin(o, key) for o in df["Origin"]]
    df.to_pickle(os.path.join(OUT_DIR, "quantifiers_haze_longitudinal.pkl"))

    # anonymous cross-sectional haze illustration set ("haze N" -> no name to hide)
    df = pd.read_pickle(os.path.join(SRC_PKL_DIR, "data_set_Q3_haze_mov_win_.pkl")).rename(columns=RENAME)
    df.to_pickle(os.path.join(OUT_DIR, "quantifiers_haze_crosssection.pkl"))

    # small internal-healthy illustration set
    df = pd.read_pickle(os.path.join(SRC_PKL_DIR, "data_set_Q3_healthy_mov_win_.pkl")).rename(columns=RENAME)
    df["Origin"] = [anonymise_origin(o, key) for o in df["Origin"]]
    df.to_pickle(os.path.join(OUT_DIR, "quantifiers_healthy_small.pkl"))

    # ------------------------------------------------------------- repeatability sets
    df = build_with_snr("data_set_Q3_rep_healthy_mov_win_", "repetabilite_Healthy")
    df["Origin"] = [anonymise_origin(o, key) for o in df["Origin"]]
    df.to_pickle(os.path.join(OUT_DIR, "quantifiers_repeatability_healthy.pkl"))

    df = build_with_snr("data_set_Q3_rep_fuchs_mov_win_", "repetabilite_FuchsDMEK")
    df["Origin"] = [anonymise_origin(o, key) for o in df["Origin"]]
    df.to_pickle(os.path.join(OUT_DIR, "quantifiers_repeatability_fuchs.pkl"))

    # ------------------------------------------------- train / test split (anonymised)
    for name in ["train_dataset", "test_dataset"]:
        df = pd.read_csv(os.path.join(HERE, "..", "Dataset", name + ".csv"))
        df["Origin"] = [anonymise_origin(o, key) for o in df["Origin"]]
        df.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False)

    print("Anonymised data assets written to", OUT_DIR)


if __name__ == "__main__":
    main()
