from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CLASSIFIED = DATA / "classified"
FIGURES = OUT / "figures"

COLORS = {
    "speech": "#8B3A3A",
    "minute": "#2F2F2F",
}


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)


def _require_columns(df: pd.DataFrame, name: str, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
        }
    )


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "speeches_raw": pd.read_csv(RAW / "speeches_raw.csv", low_memory=False),
        "minutes_raw": pd.read_csv(RAW / "minutes_raw.csv", low_memory=False),
        "speeches_verified": pd.read_csv(PROCESSED / "speeches_verified.csv", low_memory=False),
        "minutes_verified": pd.read_csv(PROCESSED / "minutes_verified.csv", low_memory=False),
        "excerpts_classified": pd.read_csv(CLASSIFIED / "excerpts_classified.csv", low_memory=False),
    }

    for key, df in data.items():
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "date" in df.columns:
            df["year"] = df["date"].dt.year

    _require_columns(data["speeches_raw"], "speeches_raw.csv", ["year", "institution"])
    _require_columns(data["minutes_raw"], "minutes_raw.csv", ["year", "institution"])
    _require_columns(data["speeches_verified"], "speeches_verified.csv", ["year", "speech_id"])
    _require_columns(data["minutes_verified"], "minutes_verified.csv", ["year", "minute_id"])
    _require_columns(
        data["excerpts_classified"],
        "excerpts_classified.csv",
        ["doc_type", "institution", "speech_id", "minute_id", "ensemble_level"],
    )

    return data


# Figure 1
def fig1_temporal_trends(data: dict[str, pd.DataFrame]) -> None:
    speeches_raw = data["speeches_raw"]
    minutes_raw = data["minutes_raw"]
    speeches_verified = data["speeches_verified"]
    minutes_verified = data["minutes_verified"]

    speech_climate_by_year = speeches_verified.groupby("year")["speech_id"].nunique()
    minute_climate_by_year = minutes_verified.groupby("year")["minute_id"].nunique()
    speech_total_by_year = speeches_raw.groupby("year").size()
    minute_total_by_year = minutes_raw.groupby("year").size()

    years = range(1997, 2025)
    speech_rates = []
    minute_rates = []

    for year in years:
        sp_climate = speech_climate_by_year.get(year, 0)
        sp_total = speech_total_by_year.get(year, 0)
        mn_climate = minute_climate_by_year.get(year, 0)
        mn_total = minute_total_by_year.get(year, 0)
        speech_rates.append(100 * sp_climate / sp_total if sp_total > 0 else 0)
        minute_rates.append(100 * mn_climate / mn_total if mn_total > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, speech_rates, color=COLORS["speech"], linewidth=2, marker="o", markersize=4, label="Speeches")
    ax.plot(years, minute_rates, color=COLORS["minute"], linewidth=2, marker="s", markersize=4, label="Minutes")

    events = {2015: "Paris Agreement\nCarney Speech", 2017: "NGFS\nFounded", 2021: "COP26\nGlasgow"}
    for year, label in events.items():
        ax.axvline(x=year, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(year, ax.get_ylim()[1] * 0.95, label, ha="center", va="top", fontsize=8, color="gray")

    ax.set_xlabel("Year")
    ax.set_ylabel("Climate Mention Rate (%)")
    ax.set_title("Climate Communication Over Time: Speeches vs Minutes")
    ax.legend(loc="upper left")
    ax.set_xlim(1997, 2024)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig1_temporal_trends.pdf", bbox_inches="tight")
    plt.close(fig)


# Figure 11
def fig11_temporal_commitment(data: dict[str, pd.DataFrame]) -> None:
    excerpts = data["excerpts_classified"]

    years = range(2000, 2025)
    speech_levels = []
    minute_levels = []

    for year in years:
        sp = excerpts[(excerpts["doc_type"] == "speech") & (excerpts["year"] == year)]["ensemble_level"]
        mn = excerpts[(excerpts["doc_type"] == "minute") & (excerpts["year"] == year)]["ensemble_level"]
        speech_levels.append(sp.mean() if len(sp) > 0 else np.nan)
        minute_levels.append(mn.mean() if len(mn) > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, speech_levels, color=COLORS["speech"], linewidth=2, marker="o", markersize=4, label="Speeches")
    ax.plot(years, minute_levels, color=COLORS["minute"], linewidth=2, marker="s", markersize=4, label="Minutes")

    ax.axhline(y=1.5, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=2.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Commitment Level")
    ax.set_title("Commitment Level Trends Over Time")
    ax.legend(loc="upper left")
    ax.set_ylim(1, 3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig11_temporal_commitment.pdf", bbox_inches="tight")
    plt.close(fig)


# Figure 6
def fig6_lagarde_effect(data: dict[str, pd.DataFrame]) -> None:
    excerpts = data["excerpts_classified"]
    speeches_raw = data["speeches_raw"]
    minutes_raw = data["minutes_raw"]

    ecb_excerpts = excerpts[excerpts["institution"] == "European Central Bank"]
    ecb_speeches_raw = speeches_raw[speeches_raw["institution"] == "European Central Bank"]
    ecb_minutes_raw = minutes_raw[minutes_raw["institution"] == "European Central Bank"]

    lagarde_start = 2019 + 10 / 12

    years = range(2010, 2025)
    speech_rates = []
    minute_rates = []
    speech_commitment = []
    minute_commitment = []

    for year in years:
        sp_total = len(ecb_speeches_raw[ecb_speeches_raw["year"] == year])
        sp_climate = ecb_excerpts[(ecb_excerpts["doc_type"] == "speech") & (ecb_excerpts["year"] == year)]["speech_id"].nunique()
        speech_rates.append(100 * sp_climate / sp_total if sp_total > 0 else np.nan)

        mn_total = len(ecb_minutes_raw[ecb_minutes_raw["year"] == year])
        mn_climate = ecb_excerpts[(ecb_excerpts["doc_type"] == "minute") & (ecb_excerpts["year"] == year)]["minute_id"].nunique()
        minute_rates.append(100 * mn_climate / mn_total if mn_total > 0 else np.nan)

        sp_level = ecb_excerpts[(ecb_excerpts["doc_type"] == "speech") & (ecb_excerpts["year"] == year)]["ensemble_level"].mean()
        mn_level = ecb_excerpts[(ecb_excerpts["doc_type"] == "minute") & (ecb_excerpts["year"] == year)]["ensemble_level"].mean()
        speech_commitment.append(sp_level)
        minute_commitment.append(mn_level)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(years, speech_rates, color=COLORS["speech"], linewidth=2, marker="o", markersize=5, label="Speeches")
    ax1.plot(years, minute_rates, color=COLORS["minute"], linewidth=2, marker="s", markersize=5, label="Minutes")
    ax1.axvline(x=lagarde_start, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.text(lagarde_start - 0.1, ax1.get_ylim()[1] * 0.95, "Lagarde\n Starts", ha="right", va="top", fontsize=9, color="gray")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Climate Mention Rate (%)")
    ax1.set_title("A. ECB Climate Mention Rates")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(years, speech_commitment, color=COLORS["speech"], linewidth=2, marker="o", markersize=5, label="Speeches")
    ax2.plot(years, minute_commitment, color=COLORS["minute"], linewidth=2, marker="s", markersize=5, label="Minutes")
    ax2.axvline(x=lagarde_start, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.text(lagarde_start - 0.1, 2.8, "Lagarde\n Starts", ha="right", va="top", fontsize=9, color="gray")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Mean Commitment Level")
    ax2.set_title("B. ECB Commitment Levels")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1, 3)

    plt.savefig(FIGURES / "fig6_lagarde_effect.pdf", bbox_inches="tight")
    plt.close(fig)


# Figure quality-lag a
def fig_commitment_grouped(data: dict[str, pd.DataFrame]) -> None:
    ex = data["excerpts_classified"]
    speech = ex[ex["doc_type"] == "speech"]["ensemble_level"].dropna()
    minute = ex[ex["doc_type"] == "minute"]["ensemble_level"].dropna()

    levels_numeric = [1.0, 1.5, 2.0, 2.5, 3.0]
    levels = ["1.0\n(Observ.)", "1.5", "2.0\n(Intent.)", "2.5", "3.0\n(Oper.)"]
    speech_pct = [round(100.0 * (speech == lv).mean(), 1) for lv in levels_numeric]
    minute_pct = [round(100.0 * (minute == lv).mean(), 1) for lv in levels_numeric]

    speech_n = int(len(speech))
    minute_n = int(len(minute))
    speech_mean = float(speech.mean())
    minute_mean = float(minute.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(levels))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        speech_pct,
        width,
        label=f"Speeches (n={speech_n:,})",
        color=COLORS["speech"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        minute_pct,
        width,
        label=f"Minutes (n={minute_n})",
        color=COLORS["minute"],
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, pct in zip(bars1, speech_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8, color=COLORS["speech"])

    for bar, pct in zip(bars2, minute_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8, color=COLORS["minute"])

    ax.set_xlabel("Commitment Level")
    ax.set_ylabel("Percentage of Excerpts")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 70)
    ax.set_title(f"Mean: Speeches={speech_mean:.2f}, Minutes={minute_mean:.2f}    (Mann-Whitney p=0.029)", fontsize=10, loc="left")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES / "fig_commitment_grouped.pdf", bbox_inches="tight")
    plt.close(fig)


# Figure quality-lag b
def fig_first_mention_lag(data: dict[str, pd.DataFrame]) -> None:
    ex = data["excerpts_classified"]
    speeches_raw = data["speeches_raw"].reset_index(drop=True).copy()
    minutes_raw = data["minutes_raw"].reset_index(drop=True).copy()
    speeches_raw["speech_id"] = speeches_raw.index.astype(int)
    minutes_raw["minute_id"] = minutes_raw.index.astype(int)

    speech_first = (
        ex[ex["doc_type"] == "speech"][["institution", "speech_id"]]
        .drop_duplicates()
        .merge(speeches_raw[["speech_id", "year"]], on="speech_id", how="left")
        .groupby("institution")["year"]
        .min()
    )
    minute_first = (
        ex[ex["doc_type"] == "minute"][["institution", "minute_id"]]
        .drop_duplicates()
        .merge(minutes_raw[["minute_id", "year"]], on="minute_id", how="left")
        .groupby("institution")["year"]
        .min()
    )

    common = sorted(set(speech_first.index) & set(minute_first.index))
    rows = []
    for inst in common:
        sp = speech_first.get(inst)
        mn = minute_first.get(inst)
        if pd.isna(sp) or pd.isna(mn):
            continue
        rows.append(
            {
                "institution": inst,
                "first_speech": int(sp),
                "first_minute": int(mn),
                "lag": int(mn - sp),
            }
        )

    data = sorted(rows, key=lambda x: x["lag"], reverse=True)
    mean_lag = sum(d["lag"] for d in data) / len(data)

    name_map = {
        "Federal Reserve": "Federal Reserve",
        "European Central Bank": "ECB",
        "Reserve Bank of India": "RBI",
        "Bank of Japan": "Bank of Japan",
        "Bank of Thailand": "Bank of Thailand",
        "People's Bank of China": "PBoC",
        "Sveriges Riksbank": "Riksbank",
        "Central Bank of Turkey": "CB Turkey",
        "Reserve Bank of Australia": "RBA",
        "National Bank of Romania": "NB Romania",
        "Bank of Mexico": "Bank of Mexico",
        "National Bank of Poland": "NB Poland",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(data))

    for i, d in enumerate(data):
        ax.hlines(y=i, xmin=0, xmax=d["lag"], color=COLORS["minute"], alpha=0.3, linewidth=1.5)

    lags = [d["lag"] for d in data]
    ax.scatter(lags, y_pos, color=COLORS["speech"], s=80, zorder=3, edgecolor="white", linewidth=0.5)

    for i, d in enumerate(data):
        ax.text(d["lag"] + 0.8, i, f"{d['lag']}y", va="center", ha="left", fontsize=9, color=COLORS["speech"])

    for i, d in enumerate(data):
        ax.text(-1, i, f"({d['first_speech']}-{d['first_minute']})", va="center", ha="right", fontsize=7, color="gray")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([name_map.get(d["institution"], d["institution"]) for d in data])
    ax.set_xlabel("Years Between First Speech and First Minutes Mention")

    ax.axvline(x=mean_lag, color=COLORS["speech"], linestyle="--", alpha=0.5, linewidth=1)
    ax.text(mean_lag + 0.5, len(data) - 0.5, f"Mean: {mean_lag:.1f}y", fontsize=9, color=COLORS["speech"], va="bottom")

    ax.set_xlim(-8, 35)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig_first_mention_lag.pdf", bbox_inches="tight")
    plt.close(fig)


def run_figures() -> None:
    ensure_dirs()
    setup_style()
    data = load_data()
    fig1_temporal_trends(data)
    fig11_temporal_commitment(data)
    fig6_lagarde_effect(data)
    fig_commitment_grouped(data)
    fig_first_mention_lag(data)


if __name__ == "__main__":
    run_figures()
