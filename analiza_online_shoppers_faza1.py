# KORIŠTENJE: python analiza_online_shoppers.py --csv online_shoppers_intention.csv --outdir analiza_file
import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def split_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def basic_info(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    missing = df.isna().sum()
    if missing.sum() > 0:
        miss_tbl = (missing[missing > 0].sort_values(ascending=False)).to_string()
        lines.append("Missing values (non-zero only):")
        lines.append(miss_tbl)
    else:
        lines.append("Missing values: none")
    return "\n".join(lines)


def numeric_descriptives(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"50%": "median", "25%": "25%", "75%": "75%"})
    keep = ["count", "mean", "std", "min", "25%", "median", "75%", "max"]
    desc = desc[keep]
    return desc

    


def categorical_descriptives(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    records = []
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        top_val = vc.index[0] if len(vc) else np.nan
        top_freq = int(vc.iloc[0]) if len(vc) else 0
        records.append({
            "variable": col,
            "unique": df[col].nunique(dropna=False),
            "mode": top_val,
            "mode_freq": top_freq
        })
    cat_desc = pd.DataFrame.from_records(records).set_index("variable")
    return cat_desc


def _table_to_png(df: pd.DataFrame, out_path: Path, title: str = None, max_rows: int = 25):
    show_df = df if len(df) <= max_rows else df.head(max_rows)
    fig, ax = plt.subplots(figsize=(12, min(0.4 * len(show_df) + 2, 9)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=20, pad=12)
    tbl = ax.table(cellText=np.round(show_df.values, 6) if np.issubdtype(show_df.values.dtype, np.number) else show_df.values,
                   colLabels=show_df.columns,
                   rowLabels=show_df.index,
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, numeric_cols: list, outdir: Path, bins: int = 30):
    pairs = [numeric_cols[i:i+2] for i in range(0, len(numeric_cols), 2)]
    for pair in pairs:
        fig, axes = plt.subplots(1, len(pair), figsize=(12, 4))
        if len(pair) == 1:
            axes = [axes]
        for ax, col in zip(axes, pair):
            series = df[col].dropna()
            ax.hist(series, bins=bins)
            ax.set_title(col)
            ax.set_xlabel("value")
            ax.set_ylabel("count")
        fig.suptitle("Histogrami numeričkih varijabli", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = outdir / f"hist_{'_'.join([c.replace(' ', '_') for c in pair])}.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)


def plot_categorical_bars(df: pd.DataFrame, categorical_cols: list, outdir: Path, top_n: int = 20):
    pairs = [categorical_cols[i:i+2] for i in range(0, len(categorical_cols), 2)]
    for pair in pairs:
        fig, axes = plt.subplots(1, len(pair), figsize=(12, 4))
        if len(pair) == 1:
            axes = [axes]
        for ax, col in zip(axes, pair):
            vc = df[col].value_counts().head(top_n)
            ax.bar(vc.index.astype(str), vc.values)
            ax.set_title(col)
            ax.set_ylabel("frequency")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig.suptitle("Distribucije kategorijskih varijabli (Top-N)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = outdir / f"bar_{'_'.join([c.replace(' ', '_') for c in pair])}.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)

def plot_boxplots(df: pd.DataFrame, numeric_cols: list, outdir: Path):
    pairs = [numeric_cols[i:i+2] for i in range(0, len(numeric_cols), 2)]
    
    for pair in pairs:
        fig, axes = plt.subplots(1, len(pair), figsize=(12, 5))
        if len(pair) == 1:
            axes = [axes]

        for ax, col in zip(axes, pair):
            ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
            ax.set_title(f"Boxplot: {col}")
            ax.set_ylabel("value")

        fig.suptitle("Provjera outliera - Boxplotovi", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fname = outdir / f"boxplot_{'_'.join([c.replace(' ', '_') for c in pair])}.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)

def plot_selected_boxplots(df: pd.DataFrame, cols: list, outdir: Path):
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 5))

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
        ax.set_title(f"Boxplot: {col}")
        ax.set_ylabel("value")

    fig.suptitle("Boxplotovi za odabrane varijable", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fname = outdir / f"boxplot_{'_'.join([c.replace(' ', '_') for c in cols])}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_corrected_categorical_features(df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if "Month" in df.columns:
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "June",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_counts = df["Month"].value_counts().reindex(month_order)
        month_counts = month_counts.dropna()

        axes[0].bar(month_counts.index, month_counts.values) 
        axes[0].set_title("Broj zapisa po mjesecu (kronološki)")
        axes[0].set_xlabel("Mjesec")
        axes[0].set_ylabel("Broj zapisa")

    if "VisitorType" in df.columns:
        vc = df["VisitorType"].value_counts()
        axes[1].bar(vc.index.astype(str), vc.values)
        axes[1].set_title("Distribucija tipa posjetitelja")
        axes[1].set_xlabel("VisitorType")
        axes[1].set_ylabel("Broj zapisa")
        axes[1].tick_params(axis='x', rotation=45)

    fig.suptitle("Ispravno prikazani mjeseci i tipovi posjetitelja", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(outdir / "ispravno_prikazano_Month_VisitorType.png", dpi=200)
    plt.close(fig)

  



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--outdir", default="eda_report", help="Directory to save outputs.")
    args = parser.parse_args()



    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_data(args.csv)
    plot_selected_boxplots(df, ["ProductRelated_Duration", "PageValues"], outdir)
    plot_selected_boxplots(df, ["Informational_Duration"], outdir)


    quality_report = f"""
        Broj redaka: {len(df)}
        Broj stupaca: {df.shape[1]}

        Nedostajuće vrijednosti po stupcima:
        {df.isna().sum().to_string()}

        Broj duplikata: {df.duplicated().sum()}
    """
    (outdir / "data_quality_report.txt").write_text(quality_report, encoding="utf-8")

    print("\n=== Kvaliteta podataka ===\n")
    print(quality_report)

    info_text = basic_info(df)
    (outdir / "basic_info.txt").write_text(info_text, encoding="utf-8")
    print(info_text)

    numeric_cols, categorical_cols = split_columns(df)

    if numeric_cols:
        num_desc = numeric_descriptives(df, numeric_cols)
        num_desc.to_csv(outdir / "numeric_descriptives.csv", encoding="utf-8")
        _table_to_png(num_desc, outdir / "numeric_descriptives.png", title="Numeričke varijable")

    if categorical_cols:
        cat_desc = categorical_descriptives(df, categorical_cols)
        cat_desc.to_csv(outdir / "categorical_descriptives.csv", encoding="utf-8")
        _table_to_png(cat_desc, outdir / "categorical_descriptives.png", title="Kategorijske varijable")
        # Ispravljen prikaz kategorijskih i diskretnih varijabli
        plot_corrected_categorical_features(df, outdir)


    if numeric_cols:
        plot_histograms(df, numeric_cols, outdir)
        plot_boxplots(df, numeric_cols, outdir)
    if categorical_cols:
        plot_categorical_bars(df, categorical_cols, outdir)

    corr = df[numeric_cols].corr(method="pearson")
    corr.to_csv(outdir / "correlation_matrix.csv", encoding="utf-8")

    print("\n=== Korelacijska matrica ===\n")
    print(corr)

    import seaborn as sns
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 7})
    plt.title("Korelacijska matrica numeričkih varijabli")
    plt.tight_layout()
    plt.savefig(outdir / "correlation_heatmap.png", dpi=200)
    plt.close()



if __name__ == "__main__":
    main()
