from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .analysis import run_analysis
except ImportError:
    from analysis import run_analysis

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
RAW = DATA / "raw"
CLASSIFIED = DATA / "classified"
ANALYSIS = OUT / "analysis"
TABLES = OUT / "tables"


def ensure_dirs() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


# Table 1
def table1_overview(main_numbers: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "Total documents", "speeches": main_numbers["speech_total_core_sample"], "minutes": main_numbers["minute_total"]},
            {
                "metric": "Institutions",
                "speeches": main_numbers["speech_institutions_core_sample"],
                "minutes": main_numbers["minute_institutions"],
            },
            {
                "metric": "Period",
                "speeches": f"{main_numbers['speech_period_start']}-{main_numbers['speech_period_end']}",
                "minutes": f"{main_numbers['minute_period_start']}-{main_numbers['minute_period_end']}",
            },
            {"metric": "Languages", "speeches": "Predominantly English", "minutes": f"{main_numbers['minute_languages']} languages"},
            {"metric": "Stage 1 documents", "speeches": main_numbers["stage1_speech_docs"], "minutes": main_numbers["stage1_minute_docs"]},
            {"metric": "Stage 1 excerpts", "speeches": main_numbers["stage1_speech_excerpts"], "minutes": main_numbers["stage1_minute_excerpts"]},
            {"metric": "Stage 2 verified documents", "speeches": main_numbers["verified_speech_docs"], "minutes": main_numbers["verified_minute_docs"]},
            {"metric": "Stage 2 verified excerpts", "speeches": main_numbers["verified_speech_excerpts"], "minutes": main_numbers["verified_minute_excerpts"]},
            {
                "metric": "False positive rate (%)",
                "speeches": main_numbers["speech_false_positive_rate_pct"],
                "minutes": main_numbers["minute_false_positive_rate_pct"],
            },
            {"metric": "Climate mention rate (%)", "speeches": main_numbers["speech_rate_core_denom_pct"], "minutes": main_numbers["minute_rate_pct"]},
            {
                "metric": "Institutions with climate content",
                "speeches": main_numbers["speech_institutions_with_climate_submission_rule"],
                "minutes": main_numbers["minute_institutions_with_climate"],
            },
        ]
    )


# Table 2
def table2_institution_heterogeneity() -> pd.DataFrame:
    minutes = pd.read_csv(RAW / "minutes_raw.csv", low_memory=False).reset_index(drop=True)
    ex = pd.read_csv(CLASSIFIED / "excerpts_classified.csv", low_memory=False)

    minutes["minute_id"] = minutes.index.astype(int)

    climate = ex[ex["doc_type"] == "minute"].copy()
    doc_counts = climate.groupby("institution")["minute_id"].nunique().rename("docs")
    excerpt_counts = climate.groupby("institution").size().rename("excerpts")

    climate_doc_years = (
        climate[["institution", "minute_id"]]
        .drop_duplicates()
        .merge(minutes[["minute_id", "year"]], on="minute_id", how="left")
        .groupby("institution")["year"]
        .agg(["min", "max"])
        .rename(columns={"min": "period_start", "max": "period_end"})
    )

    left = pd.concat([doc_counts, excerpt_counts, climate_doc_years], axis=1).reset_index()
    left = left.sort_values(["docs", "institution"], ascending=[False, True]).reset_index(drop=True)
    def _period_str(r: pd.Series) -> str:
        if pd.isna(r["period_start"]) or pd.isna(r["period_end"]):
            return ""
        return f"{int(r['period_start'])}-{int(r['period_end'])}"

    left["period"] = left.apply(_period_str, axis=1)

    climate_institutions = set(left["institution"])
    right = minutes[~minutes["institution"].isin(climate_institutions)].groupby("institution").size().rename("total_minutes").reset_index()
    right = right.sort_values(["total_minutes", "institution"], ascending=[False, True]).reset_index(drop=True)

    n = max(len(left), len(right))
    rows = []
    for i in range(n):
        l = left.iloc[i] if i < len(left) else None
        r = right.iloc[i] if i < len(right) else None
        rows.append(
            {
                "climate_institution": l["institution"] if l is not None else "",
                "climate_docs": int(l["docs"]) if l is not None else "",
                "climate_excerpts": int(l["excerpts"]) if l is not None else "",
                "climate_period": l["period"] if l is not None else "",
                "no_climate_institution": r["institution"] if r is not None else "",
                "no_climate_total_minutes": int(r["total_minutes"]) if r is not None else "",
            }
        )

    rows.append(
        {
            "climate_institution": "TOTAL",
            "climate_docs": int(left["docs"].sum()),
            "climate_excerpts": int(left["excerpts"].sum()),
            "climate_period": "",
            "no_climate_institution": "TOTAL",
            "no_climate_total_minutes": int(right["total_minutes"].sum()),
        }
    )

    return pd.DataFrame(rows)


def run_tables() -> None:
    ensure_dirs()
    run_analysis()
    main_numbers = json.loads((ANALYSIS / "main_numbers.json").read_text(encoding="utf-8"))

    t1 = table1_overview(main_numbers)
    t2 = table2_institution_heterogeneity()

    t1.to_csv(TABLES / "table1_overview.csv", index=False)
    t2.to_csv(TABLES / "table2_institution_heterogeneity.csv", index=False)


if __name__ == "__main__":
    run_tables()
