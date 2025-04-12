import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore

# Load data
df = pd.read_csv("C:\\Users\\tpriy\\OneDrive\\ドキュメント\\Project sem 4\\Crime_Incidents_in_2024 (2).csv")

# Preprocessing
df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'], errors='coerce')
df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')

# Remove timezone before converting to Period
df['Month'] = df['REPORT_DAT'].dt.tz_localize(None).dt.to_period('M').astype(str)
df['DayOfWeek'] = df['REPORT_DAT'].dt.day_name()
df['Hour'] = df['REPORT_DAT'].dt.hour

# Colors for manual dark theme
bg_color = '#1e1e1e'
fg_color = '#e0e0e0'
grid_color = '#444444'

def apply_dark_theme(ax):
    ax.set_facecolor(bg_color)
    ax.figure.set_facecolor(bg_color)
    ax.tick_params(colors=fg_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    ax.title.set_color(fg_color)
    ax.xaxis.label.set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)

# Temporal Analysis
fig, axs = plt.subplots(3, 1, figsize=(15, 12))

monthly = df['Month'].value_counts().sort_index()
sns.lineplot(x=monthly.index, y=monthly.values, ax=axs[0], marker='o', color='cyan')
axs[0].set_title("Monthly Crime Trends")
axs[0].set_ylabel("Number of Crimes")
axs[0].tick_params(axis='x', rotation=45)
apply_dark_theme(axs[0])

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_df = df['DayOfWeek'].value_counts().reindex(weekday_order).reset_index()
weekday_df.columns = ['Day', 'Count']
sns.barplot(data=weekday_df, x='Day', y='Count', hue='Day', legend=False, palette='pastel', ax=axs[1])
axs[1].set_title("Crimes by Day of the Week")
apply_dark_theme(axs[1])

hourly = df['Hour'].value_counts().sort_index()
sns.lineplot(x=hourly.index, y=hourly.values, ax=axs[2], marker='o', color='lime')
axs[2].set_title("Crimes by Hour of the Day")
axs[2].set_xlabel("Hour (0-23)")
axs[2].set_ylabel("Number of Crimes")
apply_dark_theme(axs[2])

plt.tight_layout()
plt.show()

# Crime Type Distribution
fig, axs = plt.subplots(2, 1, figsize=(15, 10))

top_types_df = df['OFFENSE'].value_counts().head(10).reset_index()
top_types_df.columns = ['OFFENSE', 'Count']
sns.barplot(data=top_types_df, x='Count', y='OFFENSE', hue='OFFENSE', legend=False, palette='coolwarm', ax=axs[0])
axs[0].set_title("Top 10 Most Frequent Crime Types")
apply_dark_theme(axs[0])

if 'METHOD' in df.columns:
    severity = df.groupby(['OFFENSE', 'METHOD']).size().unstack(fill_value=0)
    severity = severity[severity.sum(axis=1) > 50]
    severity.plot(kind='barh', stacked=True, colormap='Set2', ax=axs[1])
    axs[1].set_title("Crime Type vs Method (Severity Proxy)")
    apply_dark_theme(axs[1])
else:
    print("METHOD column missing!")

plt.tight_layout()
plt.show()

# Geographic Analysis
fig, axs = plt.subplots(2, 1, figsize=(15, 10))

if 'NEIGHBORHOOD_CLUSTER' in df.columns:
    top_hoods_df = df['NEIGHBORHOOD_CLUSTER'].value_counts().head(10).reset_index()
    top_hoods_df.columns = ['Neighborhood', 'Count']
    sns.barplot(data=top_hoods_df, x='Count', y='Neighborhood', hue='Neighborhood', legend=False, palette='magma', ax=axs[0])
    axs[0].set_title("Top 10 Neighborhoods with Most Crimes")
    apply_dark_theme(axs[0])

if 'WARD' in df.columns:
    ward_df = df['WARD'].value_counts().sort_index().reset_index()
    ward_df.columns = ['Ward', 'Count']
    sns.barplot(data=ward_df, x='Ward', y='Count', hue='Ward', legend=False, palette='cubehelix', ax=axs[1])
    axs[1].set_title("Crime Distribution by Ward")
    apply_dark_theme(axs[1])

plt.tight_layout()
plt.show()

# Law Enforcement Response
fig, axs = plt.subplots(2, 1, figsize=(15, 10))

if 'START_DATE' in df.columns and 'REPORT_DAT' in df.columns:
    df['Response_Minutes'] = (df['REPORT_DAT'] - df['START_DATE']).dt.total_seconds() / 60
    valid = df[df['Response_Minutes'] >= 0]
    sns.histplot(valid['Response_Minutes'], bins=50, color='skyblue', ax=axs[0])
    axs[0].set_title("Response Time Distribution (Minutes)")
    apply_dark_theme(axs[0])

if 'DISPOSITION' in df.columns:
    disp_df = df['DISPOSITION'].value_counts().reset_index()
    disp_df.columns = ['Disposition', 'Count']
    sns.barplot(data=disp_df, x='Disposition', y='Count', hue='Disposition', legend=False, palette='pastel', ax=axs[1])
    axs[1].set_title("Clearance Status")
    apply_dark_theme(axs[1])

plt.tight_layout()
plt.show()

# Correlation Analysis
numeric_cols = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1)
corr = numeric_cols.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix", color=fg_color)
plt.gca().set_facecolor(bg_color)
plt.gcf().set_facecolor(bg_color)
plt.xticks(color=fg_color)
plt.yticks(color=fg_color)
plt.show()

# Boxplot Outlier Detection
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols.columns[:3]):
    plt.subplot(1, 3, i+1)
    sns.boxplot(data=df, y=col, color='orange')
    plt.title(f'Boxplot of {col}', color=fg_color)
    plt.gca().set_facecolor(bg_color)
    plt.xticks(color=fg_color)
    plt.yticks(color=fg_color)
    plt.ylabel('')
plt.tight_layout()
plt.gcf().set_facecolor(bg_color)
plt.show()

# Z-score Outlier Detection
z_scores = numeric_cols.apply(zscore).abs()
outliers = (z_scores > 3).sum()
print("\nOutlier count (Z-score > 3):")
print(outliers)
