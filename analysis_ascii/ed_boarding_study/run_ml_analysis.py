from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from run_ed_boarding_study import OUTPUT_ROOT, SHARED_CORE_EXCLUDE


ML_OUTPUT_DIR = OUTPUT_ROOT / "ml"
ML_TABLE_DIR = ML_OUTPUT_DIR / "tables"
ML_FIG_DIR = ML_OUTPUT_DIR / "figures"
ML_PRED_DIR = ML_OUTPUT_DIR / "predictions"


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_val: np.ndarray
    groups_test: np.ndarray
    feature_columns: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


def ensure_dirs() -> None:
    for path in [ML_OUTPUT_DIR, ML_TABLE_DIR, ML_FIG_DIR, ML_PRED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_ROOT / "features" / "mimic_model_dataset.csv.gz", low_memory=False)


def prepare_split(df: pd.DataFrame) -> SplitData:
    outcome_col = "unexpected_icu_24h"
    groups = df["subject_id"].to_numpy()
    drop_cols = list(SHARED_CORE_EXCLUDE)

    keep = df.drop(columns=drop_cols, errors="ignore").copy()
    feature_columns = [c for c in keep.columns if c != outcome_col]
    X = keep[feature_columns].copy()
    y = df[outcome_col].astype(int).to_numpy()

    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(outer.split(X, y, groups))
    X_trainval = X.iloc[trainval_idx].reset_index(drop=True)
    y_trainval = y[trainval_idx]
    groups_trainval = groups[trainval_idx]
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    groups_test = groups[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    train_idx, val_idx = next(inner.split(X_trainval, y_trainval, groups_trainval))

    split_counts = pd.DataFrame(
        [
            {"split": "train", "n": len(train_idx), "event_rate": y_trainval[train_idx].mean()},
            {"split": "validation", "n": len(val_idx), "event_rate": y_trainval[val_idx].mean()},
            {"split": "test", "n": len(test_idx), "event_rate": y_test.mean()},
        ]
    )
    split_counts.to_csv(ML_TABLE_DIR / "ml_split_counts.csv", index=False)

    return SplitData(
        X_train=X_trainval.iloc[train_idx].reset_index(drop=True),
        y_train=y_trainval[train_idx],
        X_val=X_trainval.iloc[val_idx].reset_index(drop=True),
        y_val=y_trainval[val_idx],
        X_test=X_test,
        y_test=y_test,
        groups_train=groups_trainval[train_idx],
        groups_val=groups_trainval[val_idx],
        groups_test=groups_test,
        feature_columns=feature_columns,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(logits, y_true)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_stats(y_true, y_prob)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def fit_and_predict_transformed(model, X_train_t, y_train, X_eval_t) -> np.ndarray:
    model.fit(X_train_t, y_train)
    return model.predict_proba(X_eval_t)[:, 1]


def prepare_catboost_frame(df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        out[col] = out[col].astype("object").fillna("missing").astype(str)
    for col in numeric_cols:
        out[col] = out[col].astype(float)
    return out


def select_best_model(split: SplitData) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    preprocessor = build_preprocessor(split.numeric_cols, split.categorical_cols)
    X_train_t = preprocessor.fit_transform(split.X_train)
    X_val_t = preprocessor.transform(split.X_val)

    tuning_rows: list[dict[str, object]] = []
    trained_models: dict[str, object] = {}

    logistic_grid = [
        {"C": 0.1, "penalty": "l2"},
        {"C": 1.0, "penalty": "l2"},
        {"C": 5.0, "penalty": "l2"},
    ]
    for params in logistic_grid:
        model = LogisticRegression(max_iter=2000, solver="lbfgs", C=params["C"])
        pred = fit_and_predict_transformed(model, X_train_t, split.y_train, X_val_t)
        metrics = evaluate_predictions(split.y_val, pred)
        tuning_rows.append({"model": "logistic", **params, **metrics})

    lgbm_grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50},
        {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50},
    ]
    for params in lgbm_grid:
        model = LGBMClassifier(
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            **params,
        )
        pred = fit_and_predict_transformed(model, X_train_t, split.y_train, X_val_t)
        metrics = evaluate_predictions(split.y_val, pred)
        tuning_rows.append({"model": "lightgbm", **params, **metrics})

    xgb_grid = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "reg_lambda": 1.0},
        {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 4, "reg_lambda": 1.0},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "reg_lambda": 5.0},
    ]
    for params in xgb_grid:
        model = XGBClassifier(
            random_state=42,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            **params,
        )
        pred = fit_and_predict_transformed(model, X_train_t, split.y_train, X_val_t)
        metrics = evaluate_predictions(split.y_val, pred)
        tuning_rows.append({"model": "xgboost", **params, **metrics})

    X_train_cat = prepare_catboost_frame(split.X_train, split.categorical_cols, split.numeric_cols)
    X_val_cat = prepare_catboost_frame(split.X_val, split.categorical_cols, split.numeric_cols)
    cat_feature_idx = [X_train_cat.columns.get_loc(c) for c in split.categorical_cols]
    cat_grid = [
        {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3.0},
        {"iterations": 500, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0},
        {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 10.0},
    ]
    for params in cat_grid:
        model = CatBoostClassifier(
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
            **params,
        )
        model.fit(X_train_cat, split.y_train, cat_features=cat_feature_idx)
        pred = model.predict_proba(X_val_cat)[:, 1]
        metrics = evaluate_predictions(split.y_val, pred)
        tuning_rows.append({"model": "catboost", **params, **metrics})

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df = tuning_df.sort_values(["auprc", "auroc", "brier"], ascending=[False, False, True]).reset_index(drop=True)
    tuning_df.to_csv(ML_TABLE_DIR / "ml_tuning_results.csv", index=False)

    best_configs = tuning_df.groupby("model", as_index=False).first()
    best_configs.to_csv(ML_TABLE_DIR / "ml_best_configs.csv", index=False)

    X_trainval = pd.concat([split.X_train, split.X_val], axis=0).reset_index(drop=True)
    y_trainval = np.concatenate([split.y_train, split.y_val])
    X_test = split.X_test.copy()

    preprocessor_full = build_preprocessor(split.numeric_cols, split.categorical_cols)
    X_trainval_t = preprocessor_full.fit_transform(X_trainval)
    X_test_t = preprocessor_full.transform(X_test)

    prediction_table = pd.DataFrame({"y_true": split.y_test})
    summary_rows = []
    trained_objects: dict[str, object] = {"preprocessor": preprocessor_full}

    for _, row in best_configs.iterrows():
        model_name = row["model"]
        params = row.drop(labels=["model", "auroc", "auprc", "brier", "calibration_intercept", "calibration_slope"]).dropna().to_dict()
        if model_name == "logistic":
            params["C"] = float(params["C"])
            model = LogisticRegression(max_iter=2000, solver="lbfgs", C=params["C"])
            model.fit(X_trainval_t, y_trainval)
            pred = model.predict_proba(X_test_t)[:, 1]
        elif model_name == "lightgbm":
            params = {k: float(v) if k in {"learning_rate"} else int(v) for k, v in params.items()}
            model = LGBMClassifier(random_state=42, subsample=0.8, colsample_bytree=0.8, **params)
            model.fit(X_trainval_t, y_trainval)
            pred = model.predict_proba(X_test_t)[:, 1]
        elif model_name == "xgboost":
            cast_params = {}
            for k, v in params.items():
                cast_params[k] = float(v) if k in {"learning_rate", "reg_lambda"} else int(v)
            model = XGBClassifier(
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=4,
                **cast_params,
            )
            model.fit(X_trainval_t, y_trainval)
            pred = model.predict_proba(X_test_t)[:, 1]
        else:
            X_trainval_cat = prepare_catboost_frame(X_trainval, split.categorical_cols, split.numeric_cols)
            X_test_cat = prepare_catboost_frame(X_test, split.categorical_cols, split.numeric_cols)
            cat_idx = [X_trainval_cat.columns.get_loc(c) for c in split.categorical_cols]
            cast_params = {}
            for k, v in params.items():
                cast_params[k] = float(v) if k in {"learning_rate", "l2_leaf_reg"} else int(v)
            model = CatBoostClassifier(loss_function="Logloss", random_seed=42, verbose=False, **cast_params)
            model.fit(X_trainval_cat, y_trainval, cat_features=cat_idx)
            pred = model.predict_proba(X_test_cat)[:, 1]
        trained_objects[model_name] = model
        prediction_table[f"pred_{model_name}"] = pred
        metrics = evaluate_predictions(split.y_test, pred)
        summary_rows.append({"model": model_name, **metrics})

    summary_df = pd.DataFrame(summary_rows).sort_values(["auprc", "auroc"], ascending=False).reset_index(drop=True)
    summary_df.to_csv(ML_TABLE_DIR / "ml_model_comparison_test.csv", index=False)
    prediction_table.to_csv(ML_PRED_DIR / "ml_test_predictions_all_models.csv.gz", index=False, compression="gzip")

    meta = {
        "feature_columns": split.feature_columns,
        "numeric_cols": split.numeric_cols,
        "categorical_cols": split.categorical_cols,
        "best_model": summary_df.iloc[0]["model"],
    }
    (ML_OUTPUT_DIR / "ml_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary_df, trained_objects, prediction_table


def make_threshold_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for burden in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        threshold = np.quantile(y_prob, 1 - burden)
        alert = (y_prob >= threshold).astype(int)
        tp = int(((alert == 1) & (y_true == 1)).sum())
        fp = int(((alert == 1) & (y_true == 0)).sum())
        tn = int(((alert == 0) & (y_true == 0)).sum())
        fn = int(((alert == 0) & (y_true == 1)).sum())
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        rows.append(
            {
                "alert_burden": burden,
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "alerts": int(alert.sum()),
                "events_captured": tp,
            }
        )
    return pd.DataFrame(rows)


def make_calibration_points(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    return pd.DataFrame(
        {
            "model": model_name,
            "bin": np.arange(1, len(frac_pos) + 1),
            "mean_predicted": mean_pred,
            "observed": frac_pos,
        }
    )


def create_figures(pred_table: pd.DataFrame, comparison: pd.DataFrame) -> None:
    y_true = pred_table["y_true"].to_numpy()
    model_cols = [c for c in pred_table.columns if c.startswith("pred_")]
    display_names = {c: c.replace("pred_", "") for c in model_cols}

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for col in model_cols:
        fpr, tpr, _ = roc_curve(y_true, pred_table[col])
        roc_auc = roc_auc_score(y_true, pred_table[col])
        ax.plot(fpr, tpr, lw=2, label=f"{display_names[col]} (AUC {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#A0AEC0")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ML_FIG_DIR / "ml_roc_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for col in model_cols:
        precision, recall, _ = precision_recall_curve(y_true, pred_table[col])
        ap = average_precision_score(y_true, pred_table[col])
        ax.plot(recall, precision, lw=2, label=f"{display_names[col]} (AP {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall comparison")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ML_FIG_DIR / "ml_pr_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    calib_frames = []
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for col in model_cols:
        df = make_calibration_points(y_true, pred_table[col], display_names[col])
        calib_frames.append(df)
        ax.plot(df["mean_predicted"], df["observed"], marker="o", lw=1.8, label=display_names[col])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#A0AEC0")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration comparison")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ML_FIG_DIR / "ml_calibration_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.concat(calib_frames, ignore_index=True).to_csv(ML_TABLE_DIR / "ml_calibration_points.csv", index=False)

    best_model = comparison.sort_values(["auprc", "auroc"], ascending=False).iloc[0]["model"]
    best_col = f"pred_{best_model}"
    thresh_df = make_threshold_table(y_true, pred_table[best_col].to_numpy())
    thresh_df.to_csv(ML_TABLE_DIR / "ml_threshold_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    x = thresh_df["alert_burden"] * 100
    ax.plot(x, thresh_df["sensitivity"], marker="o", lw=2, label="Sensitivity")
    ax.plot(x, thresh_df["ppv"], marker="s", lw=2, label="PPV")
    ax.plot(x, thresh_df["npv"], marker="^", lw=2, label="NPV")
    ax.set_xlabel("Alert burden (%)")
    ax.set_ylabel("Metric")
    ax.set_title(f"Decision-threshold metrics for best model ({best_model})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ML_FIG_DIR / "ml_threshold_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def transport_best_model(trained_objects: dict[str, object]) -> None:
    metadata = json.loads((ML_OUTPUT_DIR / "ml_metadata.json").read_text(encoding="utf-8"))
    best_model_name = metadata["best_model"]
    best_model = trained_objects[best_model_name]

    mcmed = pd.read_csv(OUTPUT_ROOT / "features" / "mcmed_model_dataset.csv.gz", low_memory=False)
    feature_columns = metadata["feature_columns"]
    numeric_cols = metadata["numeric_cols"]
    categorical_cols = metadata["categorical_cols"]

    if best_model_name == "catboost":
        X_mcmed = mcmed.reindex(columns=feature_columns).copy()
        X_mcmed = prepare_catboost_frame(X_mcmed, categorical_cols, numeric_cols)
        pred = best_model.predict_proba(X_mcmed)[:, 1]
    else:
        preprocessor = trained_objects["preprocessor"]
        X_mcmed = mcmed.reindex(columns=feature_columns, fill_value=np.nan)
        X_mcmed_t = preprocessor.transform(X_mcmed)
        pred = best_model.predict_proba(X_mcmed_t)[:, 1]

    out = mcmed[["MRN", "CSN", "boarded_6h", "post_decision_boarding_minutes"]].copy()
    out["predicted_occult_critical_illness_risk"] = pred
    out.to_csv(ML_PRED_DIR / "ml_best_model_mcmed_transport.csv.gz", index=False, compression="gzip")

    summary = pd.DataFrame(
        [
            {
                "best_model": best_model_name,
                "n": len(out),
                "median_predicted_risk": out["predicted_occult_critical_illness_risk"].median(),
                "iqr_low": out["predicted_occult_critical_illness_risk"].quantile(0.25),
                "iqr_high": out["predicted_occult_critical_illness_risk"].quantile(0.75),
                "boarded_6h_rate": out["boarded_6h"].mean(),
                "median_post_decision_boarding_minutes": out["post_decision_boarding_minutes"].median(),
            }
        ]
    )
    summary.to_csv(ML_TABLE_DIR / "ml_best_model_mcmed_transport_summary.csv", index=False)


def write_catalog() -> None:
    text = """# ML Tables And Figures

Main manuscript candidates:

- Table ML1: `tables/ml_model_comparison_test.csv`
- Table ML2: `tables/ml_threshold_metrics.csv`
- Figure ML1: `figures/ml_roc_comparison.png`
- Figure ML2: `figures/ml_pr_comparison.png`
- Figure ML3: `figures/ml_calibration_comparison.png`
- Figure ML4: `figures/ml_threshold_metrics.png`

Supplement candidates:

- Table MLS1: `tables/ml_tuning_results.csv`
- Table MLS2: `tables/ml_calibration_points.csv`
- Table MLS3: `tables/ml_split_counts.csv`
- Table MLS4: `tables/ml_best_model_mcmed_transport_summary.csv`
- Prediction file: `predictions/ml_test_predictions_all_models.csv.gz`
- MC-MED transport file: `predictions/ml_best_model_mcmed_transport.csv.gz`
"""
    (ML_OUTPUT_DIR / "ml_tables_figures_catalog.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_plot_style()
    df = load_dataset()
    split = prepare_split(df)
    comparison, trained_objects, pred_table = select_best_model(split)
    create_figures(pred_table, comparison)
    transport_best_model(trained_objects)
    write_catalog()


if __name__ == "__main__":
    main()
