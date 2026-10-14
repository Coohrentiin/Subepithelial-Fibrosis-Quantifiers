"""
Reproduce the quantitative results and figures of the paper
"Quantification of corneal surgery antecedent" from the anonymised OCT
quantifier tables.

Everything the manuscript reports is derived here from the anonymised data in
``data/`` (no patient image, no patient name). Running

    python main.py

regenerates, byte-for-byte on a given machine:
  * every figure of the paper  -> outputs/figures/
  * the minimal data set behind every mean, SD, graph and model
    as flat CSV files                                    -> outputs/csv/

The pipeline is deterministic (fixed random seed 2025).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless; write files, never open a window
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
from scipy.stats import mannwhitneyu

SEED = 2025

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "outputs", "figures")
CSV = os.path.join(HERE, "outputs", "csv")

# The nine OCT quantifiers used as classifier features (in the paper's order).
# NB: "Mean" is the relative depth of the fitted bell curve.
FEATURES = ["Peak Width", "Sigma", "COV", "Mean", "MSE",
            "Area Ratio", "Alpha", "Beta", "Peak Intensity"]


def load(name):
    return pd.read_pickle(os.path.join(DATA, name + ".pkl"))


def savefig(name):
    plt.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close()


def significance_stars(p, n_variables=1, use_bonferroni=False):
    t3 = 0.001 * (1 / n_variables) if use_bonferroni else 0.001
    t2 = 0.01 * (1 / n_variables) if use_bonferroni else 0.01
    t1 = 0.05 * (1 / n_variables) if use_bonferroni else 0.05
    if p < t3:
        return "***"
    if p < t2:
        return "**"
    if p < t1:
        return "*"
    return "n.s."


# ======================================================================
# 1. Descriptive statistics + Healthy vs Fuchs distribution comparison
# ======================================================================
def descriptive_and_distributions(q_healthy, q_fuchs):
    labels = FEATURES
    n = len(labels)

    # --- CSV: per-group descriptive stats and Mann-Whitney U test -------------
    rows = []
    for feat in labels:
        h, f = q_healthy[feat], q_fuchs[feat]
        u, p = mannwhitneyu(h, f)
        rows.append({
            "Quantifier": feat,
            "Healthy_n": len(h), "Healthy_mean": h.mean(), "Healthy_std": h.std(),
            "Healthy_median": h.median(),
            "Fuchs_n": len(f), "Fuchs_mean": f.mean(), "Fuchs_std": f.std(),
            "Fuchs_median": f.median(),
            "MannWhitneyU": u, "p_value": p,
            "significance_bonferroni": significance_stars(p, n_variables=n, use_bonferroni=True),
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(CSV, "quantifier_stats_healthy_vs_fuchs.csv"), index=False)

    # --- Figure: KDE panels with significance bars ----------------------------
    N = 3
    M = len(labels) // N
    hfont = {"fontname": "Times New Roman"}
    fig, axs = plt.subplots(M, N, figsize=(N * 3, M * 3))
    sns.set_theme({"grid.color": "red"}, font="Times New Roman", style="whitegrid")
    for i in range(M):
        for j in range(N):
            if i * N + j < len(labels):
                feat = labels[i * N + j]
                dh, df_ = q_healthy[feat], q_fuchs[feat]
                _, p = mannwhitneyu(dh, df_)
                sns.kdeplot(dh, ax=axs[i, j], label="Healthy Corneas", color="#069AF3")
                sns.kdeplot(df_, ax=axs[i, j], label="Fuchs Dystrophy", color="#FC5A50")
                lines = axs[i, j].get_lines()
                ymax = max(max(lines[0].get_ydata()), max(lines[1].get_ydata()))
                xh = lines[0].get_xdata()[np.argmax(lines[0].get_ydata())]
                xf = lines[1].get_xdata()[np.argmax(lines[1].get_ydata())]
                x1, x2 = min(xh, xf), max(xh, xf)
                bar_y = ymax + 0.05 * ymax
                axs[i, j].plot([x1, x2], [bar_y, bar_y], color="black", lw=1.5)
                axs[i, j].text(x1 + (x2 - x1) / 2, bar_y * 1.008,
                               significance_stars(p, use_bonferroni=True, n_variables=n),
                               ha="center", fontsize=12, color="black", **hfont)
                axs[i, j].set_ylim(0, 1.2 * ymax)
                axs[i, j].set_title(feat, **hfont)
    axs[2, 0].set_xlim(-2, 2.5)
    plt.tight_layout()
    savefig("MannWhitneyU_Healthy_Fuchs.pdf")
    return stats


# ======================================================================
# 2. Random-forest classifier: training, cross-validation, test metrics
# ======================================================================
def train_classifier():
    train_df = pd.read_csv(os.path.join(DATA, "train_dataset.csv"))
    test_df = pd.read_csv(os.path.join(DATA, "test_dataset.csv"))

    X_train = train_df.drop(columns=["Healthy", "Origin"])
    y_train = train_df["Healthy"]
    X_test = test_df.drop(columns=["Healthy", "Origin"])
    y_test = test_df["Healthy"]

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cwd = dict(zip(np.unique(y_train), cw))

    model = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight=cwd)

    # 5-fold stratified cross-validation on the training set (weighted F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs[:, 1])
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)

    # --- CSV: cross-validation scores ----------------------------------------
    pd.DataFrame({"fold": range(1, 6), "f1_weighted": cv_scores}).to_csv(
        os.path.join(CSV, "cv_scores_random_forest.csv"), index=False)

    # --- CSV: test-set performance summary -----------------------------------
    pd.DataFrame([{
        "model": "Random Forest",
        "cv_f1_weighted_mean": cv_scores.mean(),
        "cv_f1_weighted_std": cv_scores.std(),
        "test_roc_auc": auc,
        "test_accuracy": report["accuracy"],
        "TN_fuchs_as_fuchs": cm[0, 0], "FP_fuchs_as_healthy": cm[0, 1],
        "FN_healthy_as_fuchs": cm[1, 0], "TP_healthy_as_healthy": cm[1, 1],
    }]).to_csv(os.path.join(CSV, "classification_performance.csv"), index=False)

    # --- CSV: feature importances --------------------------------------------
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        os.path.join(CSV, "feature_importances.csv"), index=False)

    # --- CSV: per-sample test predictions (anonymous) ------------------------
    preds = test_df.copy()
    preds["Prediction"] = y_pred
    preds["Prob_Fuchs"] = y_probs[:, 0]
    preds["Prob_Healthy"] = y_probs[:, 1]
    preds.to_csv(os.path.join(CSV, "test_predictions.csv"), index=False)

    # --- Figure: confusion matrix --------------------------------------------
    plt.figure()
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=False)
    plt.text(0.5, 0.5, f"True Positive\n{cm[0,0]}\n{cm[0,0]/np.sum(cm[0]):.2%}", ha="center", va="center", color="white")
    plt.text(0.5, 1.5, f"False Positive\n{cm[1,0]}\n{cm[1,0]/np.sum(cm[1]):.2%}", ha="center", va="center")
    plt.text(1.5, 0.5, f"False Negative\n{cm[0,1]}\n{cm[0,1]/np.sum(cm[0]):.2%}", ha="center", va="center")
    plt.text(1.5, 1.5, f"True Negative\n{cm[1,1]}\n{cm[1,1]/np.sum(cm[1]):.2%}", ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("Doctors Label"); plt.title("Confusion Matrix")
    plt.xticks([0.5, 1.5], ["Fuchs Distrophy", "Healthy"])
    plt.yticks([0.5, 1.5], ["Fuchs Distrophy", "Healthy"])
    savefig("Confusion_Matrix.pdf")

    # --- Figure: ROC curve ----------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right"); plt.grid()
    savefig("ROC_Curve.pdf")

    # --- Figure: feature importances -----------------------------------------
    importances = model.feature_importances_
    idx = np.argsort(importances)
    plt.figure(figsize=(5, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Feature Importances")
    plt.barh(range(len(idx)), importances[idx], color="b", align="center")
    plt.yticks(range(len(idx)), [X_train.columns[i] for i in idx])
    plt.xlabel("Relative Importance")
    savefig("Feature_Importance.pdf")
    plt.style.use("default")

    # --- Figures: probability distribution + violin --------------------------
    preds["Condition"] = preds["Healthy"].map({0: "Fuchs Dystrophy", 1: "Healthy Corneas"})
    plt.figure(figsize=(10, 6))
    sns.histplot(preds.loc[preds["Healthy"] == 0, "Prob_Fuchs"], bins=10, color="#FC5A50", label="Fuchs Distrophy")
    sns.histplot(preds.loc[preds["Healthy"] == 1, "Prob_Fuchs"], bins=10, color="#069AF3", label="Healthy Corneas")
    plt.xlabel("Fuchs Distrophy Probability"); plt.ylabel("Density")
    plt.legend(); plt.title("Distrophy Probability Distribution")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1); plt.xlim(0, 1)
    savefig("Fuchs_Probability_Distribution.pdf")

    pal = {"Fuchs Dystrophy": "#FC5A50", "Healthy Corneas": "#069AF3"}
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Condition", y="Prob_Fuchs", data=preds, palette=pal,
                   inner=None, bw=0.2, cut=0, linewidth=1, alpha=0.3)
    sns.swarmplot(x="Condition", y="Prob_Fuchs", data=preds, palette=pal,
                  alpha=0.7, size=4, dodge=True)
    plt.title("Fuchs Dystrophy Probability Distribution (Violin + Swarm Plot)")
    plt.ylabel("Fuchs Dystrophy Probability"); plt.xlabel("Condition")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1); plt.ylim(0, 1)
    savefig("Fuchs_Probability_Violin_Spread.pdf")

    return model, X_train.columns, test_df


# ======================================================================
# 3. Longitudinal post-PRK (haze) evolution
# ======================================================================
def _snr_size(series, lo, hi):
    return 40 + (500 - 40) * (series - lo) / (hi - lo)


def haze_longitudinal(model, feat_cols):
    q = load("quantifiers_haze_longitudinal")
    q["Patient_ID"] = q["Origin"].apply(lambda x: x.split("_")[0] + "_" + x.split("_")[1])

    def month(x):
        t = x.rstrip(".").split("_")[-1]
        return -1 if t == "normal" else int(t[1:])
    q["Month_postOp"] = q["Origin"].apply(month)

    q["Healthy_Probability"] = model.predict_proba(q[feat_cols])[:, 1]
    q["Dystrophy_Probability"] = model.predict_proba(q[feat_cols])[:, 0]
    q["Healthy_Predicted"] = model.predict(q[feat_cols])
    q.to_csv(os.path.join(CSV, "haze_longitudinal_predictions.csv"), index=False)

    patients = q["Patient_ID"].unique()
    npat = len(patients)
    palette = cm.get_cmap("tab10", npat)
    lo, hi = q["SNR"].min(), q["SNR"].max()

    def evolution_plot(ycol, ylabel, fname, hline=None, relative=False, only=None):
        plt.figure(figsize=(10, 8))
        for pat in patients:
            if only is not None and pat != only:
                continue
            d = q[q["Patient_ID"] == pat]
            order = np.argsort(d["Month_postOp"])
            m = d["Month_postOp"].iloc[order]
            y = d[ycol].iloc[order]
            if relative:
                y = np.array(y) - np.array(y)[0]
            s = _snr_size(d["SNR"].iloc[order], lo, hi)
            c = palette(patients.tolist().index(pat))
            plt.plot(m, y, linestyle="--", linewidth=2, color=c)
            plt.scatter(m, y, s=s, color=c, alpha=0.8)
        if hline is not None:
            plt.axhline(y=hline, color="r", linestyle="-.", linewidth=2)
        plt.axvline(x=0, color="g", linestyle="-.", linewidth=2)
        plt.xlabel("Months post-op"); plt.ylabel(ylabel)
        legend_lines = [Line2D([0], [0], linestyle="--", marker="o", color=palette(i), lw=2, markersize=10)
                        for i in range(npat)]
        plt.legend(legend_lines, patients, title="Patient ID", bbox_to_anchor=(1.05, 1), loc="upper left")
        savefig(fname)

    evolution_plot("Dystrophy_Probability", "Predicted Probability of Dystrophy",
                   "Probability_Evolution.pdf", hline=0.5)
    evolution_plot("Dystrophy_Probability", "Predicted Probability of Dystrophy",
                   "Probability_Evolution_example.pdf", hline=0.5, only=patients[0])
    evolution_plot("Peak Width", "Peak Width", "Peak_Width_Evolution.pdf")
    evolution_plot("Sigma", "Sigma", "Sigma_Evolution.pdf")
    evolution_plot("Dystrophy_Probability", "Relative Predicted Probability of dystrophy",
                   "Relative_Probability_Evolution.pdf", relative=True)
    evolution_plot("Peak Width", "Relative Peak Width",
                   "Relative_PeakWidth_Evolution.pdf", relative=True)
    return q


# ======================================================================
# 4. Repeatability (healthy corneas and post-DMEK / haze corneas)
# ======================================================================
def repeatability(model, feat_cols):
    def prep(name, id_parts):
        q = load(name)
        q["Patient_ID"] = q["Origin"].apply(lambda x: "_".join(x.split("_")[:id_parts]))
        q["Dystrophy_Probability"] = model.predict_proba(q[feat_cols])[:, 0]
        q["Healthy_Predicted"] = model.predict(q[feat_cols])
        return q

    rh = prep("quantifiers_repeatability_healthy", 1)
    rf = prep("quantifiers_repeatability_fuchs", 3)
    rh.to_csv(os.path.join(CSV, "repeatability_healthy_predictions.csv"), index=False)
    rf.to_csv(os.path.join(CSV, "repeatability_fuchs_predictions.csv"), index=False)

    def scatter_plot(q, ycol, ylabel, fname, ylim, xlabel, reverse=False, figw=12):
        patients = q["Patient_ID"].unique()
        palette = cm.get_cmap("tab20", 20)
        lo, hi = q["SNR"].min(), q["SNR"].max()
        plt.figure(figsize=(figw, 8))
        for i, pat in enumerate(patients):
            d = q[q["Patient_ID"] == pat]
            s = _snr_size(d["SNR"], lo, hi)
            k = (19 - patients.tolist().index(pat)) if reverse else patients.tolist().index(pat)
            plt.scatter([i] * len(d), d[ycol], s=s, color=palette(k), alpha=0.7)
        plt.ylim(*ylim); plt.xlabel(xlabel); plt.ylabel(ylabel)
        savefig(fname)

    scatter_plot(rh, "Dystrophy_Probability", "Predicted Probability of Dystrophy",
                 "Rep_Healthy_Probability.pdf", (0, 1), "Patient (Healthy Corneas)")
    scatter_plot(rh, "Peak Width", "Peak Width",
                 "Rep_Healthy_PeakWidth.pdf", (16, 98), "Patient (Healthy Corneas)")
    scatter_plot(rf, "Dystrophy_Probability", "Predicted Probability of Dystrophy",
                 "Rep_Haze_Probability.pdf", (0, 1), "Patient (Corneas presenting Haze)",
                 reverse=True, figw=12 / 9 * 5)
    scatter_plot(rf, "Peak Width", "Peak Width",
                 "Rep_Haze_PeakWidth.pdf", (16, 98), "Patient (Corneas presenting Haze)",
                 reverse=True, figw=12 / 9 * 5)


# ======================================================================
# 5. Three-way comparison: Fuchs vs Healthy vs Haze
# ======================================================================
def three_way(model, feat_cols, test_df):
    q_haze = load("quantifiers_haze_crosssection")
    df = pd.concat([test_df, q_haze], ignore_index=True)
    df["Dystrophy_Probability"] = model.predict_proba(df[feat_cols])[:, 0]
    df["Condition"] = df["Healthy"].map({0: "Fuchs Dystrophy", 1: "Healthy Corneas", 2: "Visual Haze"})

    pal = {"Fuchs Dystrophy": "#FC5A50", "Healthy Corneas": "#069AF3", "Visual Haze": "#a61ba9ff"}

    def violin(ycol, ylabel, fname):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Condition", y=ycol, data=df, palette=pal,
                       inner=None, bw=0.6, cut=0, linewidth=1, alpha=0.3)
        sns.swarmplot(x="Condition", y=ycol, data=df, palette=pal, alpha=0.7, size=4, dodge=True)
        plt.title(f"{ylabel} Distribution (Violin + Swarm Plot)")
        plt.ylabel(ylabel); plt.xlabel("Condition")
        plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
        savefig(fname)

    violin("Dystrophy_Probability", "Fuchs Dystrophy Probability", "Fuchs_Probability_Violin_Spread_Haze.pdf")
    violin("Peak Width", "Peak Width", "Peak_width_Violin_Spread_Haze.pdf")
    violin("Sigma", "Sigma", "Sigma_Violin_Spread_Haze.pdf")

    # --- CSV: three-way group summary (values behind the reported means/SDs) --
    rows = []
    names = {0: "Fuchs", 1: "Healthy", 2: "Haze"}
    for i, n in names.items():
        g = df[df["Healthy"] == i]
        for col in ["Dystrophy_Probability", "Peak Width", "Sigma"]:
            # np.std (population, ddof=0) to match the values reported in the paper
            rows.append({"Group": n, "Quantifier": col, "n": len(g),
                         "mean": np.mean(g[col]), "std": np.std(g[col])})
    pd.DataFrame(rows).to_csv(os.path.join(CSV, "three_way_group_summary.csv"), index=False)
    df.to_csv(os.path.join(CSV, "three_way_predictions.csv"), index=False)


# ======================================================================
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)

    q_healthy = load("quantifiers_healthy_2")
    q_fuchs = load("quantifiers_fuchs_2")

    # Concatenate the full quantifier table used for the model (anonymous).
    full = pd.concat([q_healthy, q_fuchs], ignore_index=True)
    full.to_csv(os.path.join(CSV, "quantifiers_all_corneas.csv"), index=False)

    descriptive_and_distributions(q_healthy, q_fuchs)
    model, feat_cols, test_df = train_classifier()
    haze_longitudinal(model, feat_cols)
    repeatability(model, feat_cols)
    three_way(model, feat_cols, test_df)

    print("Done.")
    print("  figures ->", FIG)
    print("  CSV     ->", CSV)


if __name__ == "__main__":
    main()
