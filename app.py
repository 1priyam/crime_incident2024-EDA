import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from scipy.stats import zscore
from pathlib import Path


st.set_page_config(page_title="Crime Incidents EDA", layout="wide")

BG_COLOR = "#1e1e1e"
FG_COLOR = "#e0e0e0"
GRID_COLOR = "#444444"
BUNDLED_DATA_PATH = Path(__file__).resolve().parent / "data" / "Crime_Incidents_in_2024.csv"


def find_candidate_csvs():
    base_dir = Path(__file__).resolve().parent
    candidates = []

    if BUNDLED_DATA_PATH.exists():
        candidates.append(BUNDLED_DATA_PATH)

    for root in [base_dir, base_dir.parent]:
        if root.exists():
            candidates.extend(sorted(root.glob("*.csv")))

    # Keep order stable and remove duplicates.
    seen = set()
    unique_candidates = []
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    return unique_candidates


def get_default_csv_path():
    if BUNDLED_DATA_PATH.exists():
        return str(BUNDLED_DATA_PATH.resolve())

    candidates = find_candidate_csvs()
    return candidates[0] if candidates else ""


def apply_dark_theme(ax):
    ax.set_facecolor(BG_COLOR)
    ax.figure.set_facecolor(BG_COLOR)
    ax.tick_params(colors=FG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.title.set_color(FG_COLOR)
    ax.xaxis.label.set_color(FG_COLOR)
    ax.yaxis.label.set_color(FG_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.2)


@st.cache_data
def load_data(uploaded_file, file_path):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path:
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Provide a CSV file to continue.")

    if "REPORT_DAT" in df.columns:
        df["REPORT_DAT"] = pd.to_datetime(df["REPORT_DAT"], errors="coerce")
        report_dates = df["REPORT_DAT"].dt.tz_localize(None)
        df["Month"] = report_dates.dt.to_period("M").astype(str)
        df["DayOfWeek"] = report_dates.dt.day_name()
        df["Hour"] = report_dates.dt.hour

    if "START_DATE" in df.columns:
        df["START_DATE"] = pd.to_datetime(df["START_DATE"], errors="coerce")

    if "START_DATE" in df.columns and "REPORT_DAT" in df.columns:
        df["Response_Minutes"] = (
            df["REPORT_DAT"] - df["START_DATE"]
        ).dt.total_seconds() / 60

    return df


def render_temporal_analysis(df):
    if not {"Month", "DayOfWeek", "Hour"}.issubset(df.columns):
        st.warning("Temporal analysis requires REPORT_DAT in the dataset.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(15, 12))

    monthly = df["Month"].value_counts().sort_index()
    sns.lineplot(x=monthly.index, y=monthly.values, ax=axs[0], marker="o", color="cyan")
    axs[0].set_title("Monthly Crime Trends")
    axs[0].set_ylabel("Number of Crimes")
    axs[0].tick_params(axis="x", rotation=45)
    apply_dark_theme(axs[0])

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_df = df["DayOfWeek"].value_counts().reindex(weekday_order).reset_index()
    weekday_df.columns = ["Day", "Count"]
    sns.barplot(
        data=weekday_df,
        x="Day",
        y="Count",
        hue="Day",
        legend=False,
        palette="pastel",
        ax=axs[1],
    )
    axs[1].set_title("Crimes by Day of the Week")
    axs[1].tick_params(axis="x", rotation=20)
    apply_dark_theme(axs[1])

    hourly = df["Hour"].value_counts().sort_index()
    sns.lineplot(x=hourly.index, y=hourly.values, ax=axs[2], marker="o", color="lime")
    axs[2].set_title("Crimes by Hour of the Day")
    axs[2].set_xlabel("Hour (0-23)")
    axs[2].set_ylabel("Number of Crimes")
    apply_dark_theme(axs[2])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_crime_distribution(df):
    if "OFFENSE" not in df.columns:
        st.warning("Crime distribution requires an OFFENSE column.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    top_types_df = df["OFFENSE"].value_counts().head(10).reset_index()
    top_types_df.columns = ["OFFENSE", "Count"]
    sns.barplot(
        data=top_types_df,
        x="Count",
        y="OFFENSE",
        hue="OFFENSE",
        legend=False,
        palette="coolwarm",
        ax=axs[0],
    )
    axs[0].set_title("Top 10 Most Frequent Crime Types")
    apply_dark_theme(axs[0])

    if "METHOD" in df.columns:
        severity = df.groupby(["OFFENSE", "METHOD"]).size().unstack(fill_value=0)
        severity = severity[severity.sum(axis=1) > 50]
        if not severity.empty:
            severity.plot(kind="barh", stacked=True, colormap="Set2", ax=axs[1])
            axs[1].set_title("Crime Type vs Method (Severity Proxy)")
            apply_dark_theme(axs[1])
        else:
            axs[1].text(0.5, 0.5, "No METHOD groups above threshold.", ha="center", va="center")
            apply_dark_theme(axs[1])
    else:
        axs[1].text(0.5, 0.5, "METHOD column missing.", ha="center", va="center")
        apply_dark_theme(axs[1])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_geographic_analysis(df):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    if "NEIGHBORHOOD_CLUSTER" in df.columns:
        top_hoods_df = df["NEIGHBORHOOD_CLUSTER"].value_counts().head(10).reset_index()
        top_hoods_df.columns = ["Neighborhood", "Count"]
        sns.barplot(
            data=top_hoods_df,
            x="Count",
            y="Neighborhood",
            hue="Neighborhood",
            legend=False,
            palette="magma",
            ax=axs[0],
        )
        axs[0].set_title("Top 10 Neighborhoods with Most Crimes")
        apply_dark_theme(axs[0])
    else:
        axs[0].text(0.5, 0.5, "NEIGHBORHOOD_CLUSTER column missing.", ha="center", va="center")
        apply_dark_theme(axs[0])

    if "WARD" in df.columns:
        ward_df = df["WARD"].value_counts().sort_index().reset_index()
        ward_df.columns = ["Ward", "Count"]
        sns.barplot(
            data=ward_df,
            x="Ward",
            y="Count",
            hue="Ward",
            legend=False,
            palette="cubehelix",
            ax=axs[1],
        )
        axs[1].set_title("Crime Distribution by Ward")
        apply_dark_theme(axs[1])
    else:
        axs[1].text(0.5, 0.5, "WARD column missing.", ha="center", va="center")
        apply_dark_theme(axs[1])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_response_analysis(df):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    if "Response_Minutes" in df.columns:
        valid = df[df["Response_Minutes"] >= 0]
        sns.histplot(valid["Response_Minutes"].dropna(), bins=50, color="skyblue", ax=axs[0])
        axs[0].set_title("Response Time Distribution (Minutes)")
        apply_dark_theme(axs[0])
    else:
        axs[0].text(0.5, 0.5, "Response time fields missing.", ha="center", va="center")
        apply_dark_theme(axs[0])

    if "DISPOSITION" in df.columns:
        disp_df = df["DISPOSITION"].value_counts().reset_index()
        disp_df.columns = ["Disposition", "Count"]
        sns.barplot(
            data=disp_df,
            x="Disposition",
            y="Count",
            hue="Disposition",
            legend=False,
            palette="pastel",
            ax=axs[1],
        )
        axs[1].set_title("Clearance Status")
        axs[1].tick_params(axis="x", rotation=20)
        apply_dark_theme(axs[1])
    else:
        axs[1].text(0.5, 0.5, "DISPOSITION column missing.", ha="center", va="center")
        apply_dark_theme(axs[1])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_correlation(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).dropna(axis=1)
    if numeric_cols.empty:
        st.warning("No numeric columns available for correlation analysis.")
        return numeric_cols

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    apply_dark_theme(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    return numeric_cols


def render_outliers(df, numeric_cols):
    if numeric_cols.empty:
        return

    fig, axes = plt.subplots(1, min(3, len(numeric_cols.columns)), figsize=(12, 6))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, col in enumerate(numeric_cols.columns[:3]):
        sns.boxplot(data=df, y=col, color="orange", ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
        axes[i].set_ylabel("")
        apply_dark_theme(axes[i])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    z_scores = numeric_cols.apply(zscore).abs()
    outliers = (z_scores > 3).sum().sort_values(ascending=False)
    st.subheader("Outlier count (Z-score > 3)")
    st.dataframe(outliers.rename("Outlier Count"))


def main():
    st.title("Crime Incidents EDA Dashboard")
    st.write("Explore crime incident data through interactive visual analysis.")

    st.sidebar.header("Data Source")
    candidate_csvs = find_candidate_csvs()
    default_csv = get_default_csv_path()
    uploaded_file = st.sidebar.file_uploader("Upload crime CSV", type=["csv"])
    selected_csv = st.sidebar.selectbox(
        "Select detected local CSV",
        options=[""] + candidate_csvs,
        index=([""] + candidate_csvs).index(default_csv) if default_csv else 0,
        format_func=lambda value: "None" if value == "" else Path(value).name,
    )
    file_path = st.sidebar.text_input("Or enter local CSV path", value=selected_csv)

    try:
        df = load_data(uploaded_file, file_path.strip())
    except Exception as exc:
        st.info("Upload a CSV file or provide a valid local path to start the dashboard.")
        st.error(str(exc))
        return

    if uploaded_file is None and file_path.strip():
        st.caption(f"Using bundled dataset: `{Path(file_path).name}`")

    st.sidebar.success(f"Loaded {len(df):,} records")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Incidents", f"{len(df):,}")
    col2.metric("Unique Offenses", f"{df['OFFENSE'].nunique() if 'OFFENSE' in df.columns else 0:,}")
    col3.metric(
        "Neighborhoods",
        f"{df['NEIGHBORHOOD_CLUSTER'].nunique() if 'NEIGHBORHOOD_CLUSTER' in df.columns else 0:,}",
    )

    st.header("Temporal Analysis")
    render_temporal_analysis(df)

    st.header("Crime Type Distribution")
    render_crime_distribution(df)

    st.header("Geographic Analysis")
    render_geographic_analysis(df)

    st.header("Law Enforcement Response")
    render_response_analysis(df)

    st.header("Correlation Analysis")
    numeric_cols = render_correlation(df)

    st.header("Outlier Detection")
    render_outliers(df, numeric_cols)


if __name__ == "__main__":
    main()
