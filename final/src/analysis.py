from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
STAGE1 = DATA / "stage1"
CLASSIFIED = DATA / "classified"
ANALYSIS = OUT / "analysis"

LAGARDE_APPOINTMENT_DATE = "2019-11-01"
LAGARDE_COUNT_WINDOW_START = "2019-07-01"

EXCLUDED_NON_CENTRAL = {
    "Bank for International Settlements",
    "International Monetary Fund",
    "Office of the Superintendent of Financial Institutions",
    "Swiss Federal Banking Commission",
}


def ensure_dirs() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)


def _require_columns(df: pd.DataFrame, name: str, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def run_analysis() -> None:
    ensure_dirs()

    speeches = pd.read_csv(RAW / "speeches_raw.csv", low_memory=False).reset_index(drop=True)
    minutes = pd.read_csv(RAW / "minutes_raw.csv", low_memory=False).reset_index(drop=True)
    stage1_s = pd.read_csv(STAGE1 / "speeches_keyword_filtered.csv", low_memory=False)
    stage1_m = pd.read_csv(STAGE1 / "minutes_keyword_filtered.csv", low_memory=False)
    speech_verified = pd.read_csv(PROCESSED / "speeches_verified.csv", low_memory=False)
    minute_verified = pd.read_csv(PROCESSED / "minutes_verified.csv", low_memory=False)
    ex = pd.read_csv(CLASSIFIED / "excerpts_classified.csv", low_memory=False)

    _require_columns(
        speeches,
        "speeches_raw.csv",
        ["year", "institution", "has_climate", "institution_harmonised"],
    )
    _require_columns(
        minutes,
        "minutes_raw.csv",
        ["Date", "year", "institution", "Language", "has_climate"],
    )
    _require_columns(stage1_s, "speeches_keyword_filtered.csv", ["speech_id"])
    _require_columns(stage1_m, "minutes_keyword_filtered.csv", ["minute_id"])
    _require_columns(
        ex,
        "excerpts_classified.csv",
        ["doc_type", "speech_id", "minute_id", "institution", "ensemble_level", "openai_level"],
    )

    speeches["speech_id"] = speeches.index.astype(int)
    minutes["minute_id"] = minutes.index.astype(int)

    speeches_core = speeches[~speeches["institution_harmonised"].isin(EXCLUDED_NON_CENTRAL)].copy()

    speech_ids_verified = set(ex.loc[ex["doc_type"] == "speech", "speech_id"].dropna().astype(int))
    minute_ids_verified = set(ex.loc[ex["doc_type"] == "minute", "minute_id"].dropna().astype(int))

    verified_speech_docs = int(ex.loc[ex["doc_type"] == "speech", "speech_id"].nunique())
    verified_minute_docs = int(ex.loc[ex["doc_type"] == "minute", "minute_id"].nunique())

    paired_rows = []
    for inst in sorted(minutes["institution"].dropna().unique()):
        s_sub = speeches[speeches["institution"] == inst]
        m_sub = minutes[minutes["institution"] == inst]
        if len(s_sub) == 0 or len(m_sub) == 0:
            continue
        paired_rows.append(
            {
                "institution": inst,
                "speech_rate": 100.0 * s_sub["speech_id"].isin(speech_ids_verified).sum() / len(s_sub),
                "minute_rate": 100.0 * m_sub["minute_id"].isin(minute_ids_verified).sum() / len(m_sub),
            }
        )
    paired_df = pd.DataFrame(paired_rows)
    t_res = stats.ttest_rel(paired_df["speech_rate"], paired_df["minute_rate"])

    speech_levels = ex.loc[ex["doc_type"] == "speech", "ensemble_level"].dropna()
    minute_levels = ex.loc[ex["doc_type"] == "minute", "ensemble_level"].dropna()
    u_stat, u_p = stats.mannwhitneyu(speech_levels, minute_levels, alternative="two-sided")

    inst_submission = ex.loc[ex["doc_type"] == "speech", "institution"].copy()
    inst_submission = inst_submission.replace({"Board of Governors of the Federal Reserve": "Federal Reserve"})

    ecb = minutes[minutes["institution"] == "European Central Bank"].copy()
    ecb["date_dt"] = pd.to_datetime(ecb["Date"], errors="coerce")
    ecb_lagarde_main = ecb[ecb["date_dt"] >= pd.Timestamp(LAGARDE_COUNT_WINDOW_START)]
    ecb_lagarde_appointment_window = ecb[ecb["date_dt"] >= pd.Timestamp(LAGARDE_APPOINTMENT_DATE)]
    ecb_ids = set(
        ex.loc[(ex["doc_type"] == "minute") & (ex["institution"] == "European Central Bank"), "minute_id"]
        .dropna()
        .astype(int)
    )

    main_numbers = {
        "speech_total_core_sample": int(len(speeches_core)),
        "speech_institutions_core_sample": int(speeches_core["institution_harmonised"].nunique()),
        "minute_total": int(len(minutes)),
        "minute_institutions": int(minutes["institution"].nunique()),
        "speech_period_start": int(speeches["year"].min()),
        "speech_period_end": int(speeches["year"].max()),
        "minute_period_start": int(minutes["year"].min()),
        "minute_period_end": int(minutes["year"].max()),
        "minute_languages": int(minutes["Language"].nunique()),
        "stage0_speech_docs": int(speeches["has_climate"].fillna(False).astype(bool).sum()),
        "stage0_minute_docs": int(minutes["has_climate"].fillna(False).astype(bool).sum()),
        "stage1_speech_docs": int(stage1_s["speech_id"].nunique()),
        "stage1_minute_docs": int(stage1_m["minute_id"].nunique()),
        "stage1_speech_excerpts": int(len(stage1_s)),
        "stage1_minute_excerpts": int(len(stage1_m)),
        "verified_speech_docs": verified_speech_docs,
        "verified_minute_docs": verified_minute_docs,
        "verified_speech_excerpts": int((ex["doc_type"] == "speech").sum()),
        "verified_minute_excerpts": int((ex["doc_type"] == "minute").sum()),
        "speech_false_positive_rate_pct": round(100.0 * (1 - verified_speech_docs / stage1_s["speech_id"].nunique()), 1),
        "minute_false_positive_rate_pct": round(100.0 * (1 - verified_minute_docs / stage1_m["minute_id"].nunique()), 1),
        "speech_rate_core_denom_pct": round(100.0 * verified_speech_docs / len(speeches_core), 1),
        "minute_rate_pct": round(100.0 * verified_minute_docs / len(minutes), 1),
        "gap_ratio": round((verified_speech_docs / len(speeches_core)) / (verified_minute_docs / len(minutes)), 1),
        "within_inst_speech_rate_pct": round(paired_df["speech_rate"].mean(), 1),
        "within_inst_minute_rate_pct": round(paired_df["minute_rate"].mean(), 1),
        "paired_t_stat": round(float(t_res.statistic), 2),
        "paired_t_p": round(float(t_res.pvalue), 3),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": round(float(u_p), 3),
        "speech_institutions_with_climate_submission_rule": int(inst_submission.nunique()),
        "minute_institutions_with_climate": int(ex.loc[ex["doc_type"] == "minute", "institution"].nunique()),
        "ecb_lagarde_docs": int(ecb_lagarde_main["minute_id"].isin(ecb_ids).sum()),
        "ecb_lagarde_total": int(len(ecb_lagarde_main)),
        "ecb_lagarde_rate_pct": round(100.0 * ecb_lagarde_main["minute_id"].isin(ecb_ids).sum() / len(ecb_lagarde_main), 1),
        "ecb_lagarde_count_window_start": LAGARDE_COUNT_WINDOW_START,
        "ecb_lagarde_appointment_date_marker": LAGARDE_APPOINTMENT_DATE,
        "ecb_lagarde_total_if_strict_appointment_window": int(len(ecb_lagarde_appointment_window)),
        "ecb_lagarde_rate_if_strict_appointment_window_pct": round(
            100.0 * ecb_lagarde_appointment_window["minute_id"].isin(ecb_ids).sum() / len(ecb_lagarde_appointment_window), 1
        ),
        "llm_openai_labels": int(ex["openai_level"].notna().sum()),
        "llm_ensemble_labels": int(ex["ensemble_level"].notna().sum()),
        "speech_verified_excerpts_file_rows": int(len(speech_verified)),
        "minute_verified_excerpts_file_rows": int(len(minute_verified)),
    }

    with open(ANALYSIS / "main_numbers.json", "w", encoding="utf-8") as f:
        json.dump(main_numbers, f, indent=2, sort_keys=True)

    paired_df.to_csv(ANALYSIS / "within_institution_rates.csv", index=False)


if __name__ == "__main__":
    run_analysis()
