from pathlib import Path

import pandas as pd

from run_ed_boarding_study import OUTPUT_ROOT, aipw_crossfit


def main() -> None:
    dml = pd.read_csv(OUTPUT_ROOT / "features" / "mimic_dml_dataset.csv.gz", parse_dates=["first_icu_intime", "first_post_ed_intime"])
    cohort = pd.read_csv(OUTPUT_ROOT / "cohort" / "mimic_base_cohort.csv.gz", parse_dates=["first_icu_intime", "first_post_ed_intime"])

    avoid_patterns = [
        "observation",
        "emergency department",
        "discharge lounge",
        "pacu",
        "intensive care",
        "icu",
        "coronary care",
        "ccu",
    ]

    ward_mask = dml["first_post_ed_careunit"].fillna("").str.lower().apply(
        lambda x: x != "" and not any(pattern in x for pattern in avoid_patterns)
    )
    ward_df = dml.loc[ward_mask].copy()
    ward_df["icu_24h_after_ward"] = (
        ward_df["first_icu_intime"].notna()
        & (ward_df["first_icu_intime"] > ward_df["first_post_ed_intime"])
        & (ward_df["first_icu_intime"] <= ward_df["first_post_ed_intime"] + pd.Timedelta(hours=24))
    ).astype(int)

    sensitivity_rows = []

    res_ward_outcome = aipw_crossfit(ward_df, "boarded_6h", "icu_24h_after_ward", "subject_id")
    res_ward_outcome["analysis"] = "direct_to_ward__icu_24h_after_ward"
    sensitivity_rows.append(res_ward_outcome)

    non_obs_mask = ~dml["first_post_ed_careunit"].fillna("").str.lower().str.contains("observation")
    non_obs_df = dml.loc[non_obs_mask].copy()
    res_non_obs = aipw_crossfit(non_obs_df, "boarded_6h", "unexpected_icu_24h_post_lm", "subject_id")
    res_non_obs["analysis"] = "exclude_observation_first_location"
    sensitivity_rows.append(res_non_obs)

    out = pd.concat(sensitivity_rows, ignore_index=True)
    out.to_csv(OUTPUT_ROOT / "causal" / "mimic_dml_sensitivity_summary.csv", index=False)

    cohort["direct_to_ward"] = cohort["first_post_ed_careunit"].fillna("").str.lower().apply(
        lambda x: x != "" and not any(pattern in x for pattern in avoid_patterns)
    )

    counts = {
        "descriptive_direct_ward_n": int((cohort["direct_to_ward"] & (cohort["direct_icu_flag"] == 0)).sum()),
        "base_direct_ward_n": int(ward_mask.sum()),
        "base_non_observation_n": int(non_obs_mask.sum()),
        "base_direct_ward_event_rate": float(ward_df["icu_24h_after_ward"].mean()),
        "base_non_observation_event_rate": float(non_obs_df["unexpected_icu_24h_post_lm"].mean()),
    }
    counts["direct_ward_excluded_after_feature_assembly"] = counts["descriptive_direct_ward_n"] - counts["base_direct_ward_n"]
    pd.DataFrame([counts]).to_csv(OUTPUT_ROOT / "causal" / "mimic_dml_sensitivity_counts.csv", index=False)

    crude = (
        ward_df
        .groupby("boarded_6h")["icu_24h_after_ward"]
        .agg(["count", "mean", "sum"])
        .reset_index()
    )
    crude["cohort"] = "analytic_direct_to_ward"
    crude = crude[["cohort", "boarded_6h", "count", "mean", "sum"]]
    crude.to_csv(OUTPUT_ROOT / "causal" / "mimic_direct_to_ward_crude_rates.csv", index=False)

    descriptive_crude = (
        cohort.loc[cohort["direct_to_ward"] & (cohort["direct_icu_flag"] == 0)]
        .assign(
            boarded_6h=lambda x: (pd.to_datetime(x["ed_outtime"]) > pd.to_datetime(x["ed_intime"]) + pd.Timedelta(hours=6)).astype(int),
            icu_24h_after_ward=lambda x: (
                x["first_icu_intime"].notna()
                & (x["first_icu_intime"] > x["first_post_ed_intime"])
                & (x["first_icu_intime"] <= x["first_post_ed_intime"] + pd.Timedelta(hours=24))
            ).astype(int),
        )
        .groupby("boarded_6h")["icu_24h_after_ward"]
        .agg(["count", "mean", "sum"])
        .reset_index()
    )
    descriptive_crude["cohort"] = "descriptive_direct_to_ward"
    descriptive_crude = descriptive_crude[["cohort", "boarded_6h", "count", "mean", "sum"]]
    descriptive_crude.to_csv(OUTPUT_ROOT / "causal" / "mimic_direct_to_ward_crude_rates_descriptive.csv", index=False)


if __name__ == "__main__":
    main()
