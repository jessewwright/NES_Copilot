"""Analysis script for reaction time speed-up effect.

This script merges Roberts et al. trial data with subject-level
NES parameter estimates and fits a mixed-effects model
examining whether higher effective norm weights predict
faster reaction times for sure choices in loss frames.

Usage:
    python -m nes_copilot.rt_speedup_analysis \
        --trial_data ftp_osf_data.csv \
        --subject_params combined_analysis_data.csv

The output is an interaction plot saved as
'rt_interaction_sure_choices.png'.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


def load_and_merge(trial_path: str, subject_path: str) -> pd.DataFrame:
    """Load trial-level and subject-level data and merge on subject ID."""
    trials = pd.read_csv(trial_path)
    subjects = pd.read_csv(subject_path)

    # ensure consistent column names
    trials.rename(columns={"subject": "subject_id"}, inplace=True)

    if "subject" in subjects.columns:
        subjects.rename(columns={"subject": "subject_id"}, inplace=True)

    merged = pd.merge(trials, subjects, on="subject_id", how="inner")
    return merged


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and create variables for analysis."""
    df = df.copy()

    # filter target trials
    df = df[df["trialType"] == "target"]

    # remove missing RTs
    df = df[df["rt"].notna()]

    # apply RT bounds
    df = df[(df["rt"] >= 0.2) & (df["rt"] <= 4.0)]

    # log transform
    df["log_rt"] = np.log(df["rt"])

    # frame indicator
    df["frame_loss"] = np.where(df["frame"] == "loss", 1, 0)

    # choice indicator
    df["choice_sure"] = np.where(df["choice"] == 0, 1, 0)

    # effective norm weight
    df["wn_eff_sbj"] = df["v_norm_mean"]
    df["wn_eff_centered"] = df["wn_eff_sbj"] - df["wn_eff_sbj"].mean()

    # time constraint
    df["cond_tc"] = np.where(df["cond"] == "tc", 1, 0)

    # keep only sure choices
    df = df[df["choice_sure"] == 1]

    return df


def fit_mixedlm(df: pd.DataFrame):
    """Fit linear mixed effects model."""
    model = smf.mixedlm(
        "log_rt ~ frame_loss * wn_eff_centered + cond_tc",
        data=df,
        groups=df["subject_id"],
        re_formula="1 + frame_loss",
    )
    result = model.fit(method="lbfgs")
    return result


def plot_interaction(df: pd.DataFrame, output_path: str) -> None:
    """Generate interaction plot with median split on wn_eff."""
    df = df.copy()
    median_val = df["wn_eff_sbj"].median()
    df["wn_group"] = np.where(df["wn_eff_sbj"] >= median_val, "High wn_eff", "Low wn_eff")

    summary = (
        df.groupby(["cond_tc", "wn_group", "frame_loss"]) ["log_rt"].mean().reset_index()
    )
    summary["frame_type"] = np.where(summary["frame_loss"] == 1, "Loss", "Gain")
    summary["cond_label"] = np.where(summary["cond_tc"] == 1, "TC", "NTC")

    sns.set(style="whitegrid")
    g = sns.catplot(
        x="frame_type",
        y="log_rt",
        hue="wn_group",
        col="cond_label",
        data=summary,
        kind="point",
        dodge=True,
        height=4,
        aspect=1,
    )
    g.set_axis_labels("Frame", "Mean log RT")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("RT interaction for sure choices")
    plt.savefig(output_path)
    plt.close()


def main(args: Tuple[str, str]):
    trial_path, subject_path = args

    if not os.path.exists(trial_path):
        raise FileNotFoundError(f"Trial data not found: {trial_path}")
    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Subject data not found: {subject_path}")

    merged = load_and_merge(trial_path, subject_path)
    prepared = prepare_data(merged)
    result = fit_mixedlm(prepared)

    print(result.summary())

    plot_interaction(prepared, "rt_interaction_sure_choices.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT speed-up analysis")
    parser.add_argument("--trial_data", required=True, help="Path to ftp_osf_data.csv")
    parser.add_argument(
        "--subject_params", required=True, help="Path to combined_analysis_data.csv"
    )
    args = parser.parse_args()
    main((args.trial_data, args.subject_params))
