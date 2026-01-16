from __future__ import annotations

import argparse
import os

import pandas as pd

from .cleaning import clean_all
from .config import Config
from .eda import ensure_dir, run_eda
from .features import add_features
from .io import load_data
from .modeling import (
    build_preprocessor,
    evaluate_and_select_model,
    train_final_and_predict,
)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end movie revenue category prediction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Folder containing train/test/sample_submission CSVs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Folder to write artifacts and submission",
    )
    parser.add_argument(
        "--submission_name",
        type=str,
        default="submissions.csv",
        help="Output submission filename",
    )
    args = parser.parse_args()

    cfg = Config()
    ensure_dir(args.out_dir)

    # 1) Load
    train, test, _ = load_data(args.data_dir, cfg)
    print("Loaded data successfully.")

    # 2) EDA artifacts (small but useful)
    run_eda(train, args.out_dir, cfg)
    print(f"EDA artifacts saved to: {args.out_dir}")

    # 3) Clean
    train = clean_all(train)
    test = clean_all(test)

    # 4) Features
    train = add_features(train)
    test = add_features(test)

    # 5) Prepare X/y
    y = train[cfg.target_col].map({"Low": 0, "High": 1}).astype(int).values

    X = train.drop(columns=[cfg.target_col]).copy()
    X_test = test.copy()

    test_titles = X_test[cfg.id_col].copy()

    # Drop identifier and raw dates (after feature extraction)
    X = X.drop(columns=[cfg.id_col])
    X_test = X_test.drop(columns=[cfg.id_col])

    for dcol in ["release_date", "dvd_release_date"]:
        if dcol in X.columns:
            X = X.drop(columns=[dcol])
        if dcol in X_test.columns:
            X_test = X_test.drop(columns=[dcol])

    print("Model input shapes:", X.shape, X_test.shape)

    # 6) Preprocessor + model selection
    preprocessor = build_preprocessor(X)
    best_model_name, _ = evaluate_and_select_model(X, y, preprocessor, cfg)

    # 7) Final train + predict
    pred = train_final_and_predict(X, y, X_test, preprocessor, best_model_name, cfg)
    pred_labels = pd.Series(pred).map({0: "Low", 1: "High"})

    # 8) Submission
    submission = pd.DataFrame({cfg.id_col: test_titles, cfg.target_col: pred_labels})

    out_path = os.path.join(args.out_dir, args.submission_name)
    submission.to_csv(out_path, index=False)

    # Also write a copy to project root for convenience (assessment-friendly)
    submission.to_csv("submissions.csv", index=False)

    print(f"✅ Submission saved to: {out_path}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
