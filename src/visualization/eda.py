"""
EDA visualisations for the Sleman poverty dataset.

Outputs:
  - outputs/eda/target_distribution.png
  - outputs/eda/target_map.png
  - outputs/eda/feature_histograms.png
"""

from math import ceil
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.config import (
    ADMIN_DESA_GEOJSON,
    ADMIN_KEC_COL,
    KECAMATAN_COL,
    LAT_COL,
    LONG_COL,
    MODEL_DATA_CENTROID_PATH,
    MODEL_DATA_PATH,
    OUTPUT_DIR,
    TARGET_COL,
)


EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"
TARGET_DIST_PATH = EDA_OUTPUT_DIR / "target_distribution.png"
TARGET_MAP_PATH = EDA_OUTPUT_DIR / "target_map.png"
FEATURE_HIST_PATH = EDA_OUTPUT_DIR / "feature_histograms.png"


def load_data() -> pd.DataFrame:
    """Load final dataset and normalise numeric columns."""
    data_path = MODEL_DATA_CENTROID_PATH if MODEL_DATA_CENTROID_PATH.exists() else MODEL_DATA_PATH
    df = pd.read_csv(data_path)

    if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    for col in df.columns:
        if col == KECAMATAN_COL:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan.")

    return df


def as_percent(series: pd.Series) -> pd.Series:
    """Convert target proportion to percent when values are in 0-1 scale."""
    clean = series.dropna()
    if len(clean) > 0 and clean.max() <= 1:
        return series * 100
    return series


def normalise_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def plot_target_distribution(df: pd.DataFrame) -> None:
    target_pct = as_percent(df[TARGET_COL])
    plot_df = df[[KECAMATAN_COL]].copy()
    plot_df["target_pct"] = target_pct
    plot_df = plot_df.sort_values("target_pct", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    axes[0].hist(
        plot_df["target_pct"].dropna(),
        bins=min(8, max(3, plot_df["target_pct"].nunique())),
        color="#2f7d8c",
        edgecolor="white",
        linewidth=1.2,
    )
    axes[0].axvline(
        plot_df["target_pct"].mean(),
        color="#d1495b",
        linewidth=2,
        linestyle="--",
        label=f"Mean: {plot_df['target_pct'].mean():.2f}%",
    )
    axes[0].set_title("Distribusi Target", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Persentase penduduk miskin (%)")
    axes[0].set_ylabel("Jumlah kecamatan")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    colors = plt.cm.YlOrRd(Normalize(plot_df["target_pct"].min(), plot_df["target_pct"].max())(plot_df["target_pct"]))
    axes[1].barh(plot_df[KECAMATAN_COL], plot_df["target_pct"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("Target per Kecamatan", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Persentase penduduk miskin (%)")
    axes[1].grid(axis="x", alpha=0.25)

    for index, value in enumerate(plot_df["target_pct"]):
        axes[1].text(value + 0.08, index, f"{value:.2f}%", va="center", fontsize=8)

    fig.suptitle("EDA Target: Persentase Penduduk Miskin", fontsize=15, fontweight="bold")
    fig.savefig(TARGET_DIST_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_target_map(df: pd.DataFrame) -> None:
    if not ADMIN_DESA_GEOJSON.exists():
        raise FileNotFoundError(f"GeoJSON tidak ditemukan: {ADMIN_DESA_GEOJSON}")

    gdf = gpd.read_file(ADMIN_DESA_GEOJSON)
    gdf_kec = gdf.dissolve(by=ADMIN_KEC_COL).reset_index()
    if gdf_kec.crs is not None:
        gdf_kec = gdf_kec.to_crs(epsg=4326)

    map_df = df[[KECAMATAN_COL, TARGET_COL]].copy()
    map_df["target_pct"] = as_percent(map_df[TARGET_COL])
    map_df["_key"] = normalise_name(map_df[KECAMATAN_COL])
    gdf_kec["_key"] = normalise_name(gdf_kec[ADMIN_KEC_COL])

    gdf_map = gdf_kec.merge(map_df, on="_key", how="left")
    missing = gdf_map[gdf_map["target_pct"].isna()][ADMIN_KEC_COL].tolist()
    if missing:
        print(f"WARNING: Kecamatan tanpa nilai target di peta: {missing}")

    vmin = gdf_map["target_pct"].min()
    vmax = gdf_map["target_pct"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd

    fig, ax = plt.subplots(figsize=(9, 10), constrained_layout=True)
    gdf_map.plot(
        column="target_pct",
        cmap=cmap,
        edgecolor="#303030",
        linewidth=0.8,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        legend=False,
        missing_kwds={"color": "#eeeeee", "edgecolor": "#777777", "hatch": "///"},
    )

    for _, row in gdf_map.iterrows():
        if pd.isna(row["target_pct"]):
            continue
        point = row.geometry.representative_point()
        ax.annotate(
            f"{row[ADMIN_KEC_COL]}\n{row['target_pct']:.2f}%",
            xy=(point.x, point.y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#666666",
                "linewidth": 0.4,
                "alpha": 0.82,
            },
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.035)
    cbar.set_label("Persentase penduduk miskin (%)")

    ax.set_title("Map Chart Target Kemiskinan per Kecamatan", fontsize=14, fontweight="bold", pad=12)
    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.savefig(TARGET_MAP_PATH, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_histograms(df: pd.DataFrame) -> None:
    excluded_cols = {TARGET_COL, LONG_COL, LAT_COL}
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in excluded_cols
    ]
    if not feature_cols:
        raise ValueError("Tidak ada fitur numerik untuk dibuat histogram.")

    ncols = 4
    nrows = ceil(len(feature_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(4, nrows * 3.2)), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    palette = [
        "#2f7d8c",
        "#d1495b",
        "#edae49",
        "#4f6d7a",
        "#7a9e7e",
        "#b07bac",
    ]

    for index, col in enumerate(feature_cols):
        ax = axes[index]
        values = df[col].dropna()
        ax.hist(
            values,
            bins=min(8, max(3, values.nunique())),
            color=palette[index % len(palette)],
            edgecolor="white",
            linewidth=1.0,
        )
        ax.axvline(values.mean(), color="#222222", linestyle="--", linewidth=1.4)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("Frekuensi")
        ax.grid(axis="y", alpha=0.22)

    for ax in axes[len(feature_cols):]:
        ax.axis("off")

    fig.suptitle("Histogram Fitur Numerik", fontsize=15, fontweight="bold")
    fig.savefig(FEATURE_HIST_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print(f"Data loaded: {len(df)} baris, {len(df.columns)} kolom")
    print(f"Target: {TARGET_COL}")

    plot_target_distribution(df)
    print(f"Target distribution saved: {TARGET_DIST_PATH}")

    plot_target_map(df)
    print(f"Target map saved: {TARGET_MAP_PATH}")

    plot_feature_histograms(df)
    print(f"Feature histograms saved: {FEATURE_HIST_PATH}")


if __name__ == "__main__":
    main()
