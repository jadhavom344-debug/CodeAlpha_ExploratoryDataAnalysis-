"""
CodeAlpha Data Analytics Internship — Task 2
Exploratory Data Analysis (EDA) on the Titanic Dataset

Author: Om Jadhav
Description:
    This script loads the Titanic passenger dataset, cleans it, and
    performs exploratory data analysis using pandas, matplotlib and
    seaborn. It prints summary statistics to the console and saves
    six charts (PNG files) that visualize survival patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
df = pd.read_csv("titanic_sample.csv")

print("=" * 60)
print("TITANIC DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumn info:")
print(df.info())

# ---------------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------------
print("\nMissing values before cleaning:\n", df.isnull().sum())

# Fill missing Age with the median age (robust to outliers)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with the most frequent port
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing values after cleaning:\n", df.isnull().sum())

# ---------------------------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------------------------
print("\nSummary statistics (numeric columns):\n", df.describe())

overall_survival_rate = df["Survived"].mean() * 100
print(f"\nOverall survival rate: {overall_survival_rate:.2f}%")

print("\nSurvival rate by sex:\n", df.groupby("Sex")["Survived"].mean() * 100)
print("\nSurvival rate by class:\n", df.groupby("Pclass")["Survived"].mean() * 100)

# ---------------------------------------------------------------------------
# 4. Chart 1 — Overall survival count
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
ax = sns.countplot(data=df, x="Survived", hue="Survived", palette=["#e74c3c", "#2ecc71"], legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Did not survive", "Survived"])
plt.title("Overall Survival Count")
plt.xlabel("")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("chart1_survival_count.png")
plt.close()

# ---------------------------------------------------------------------------
# 5. Chart 2 — Survival by gender
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="Sex", hue="Survived", palette=["#e74c3c", "#2ecc71"])
plt.title("Survival Count by Gender")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("chart2_survival_by_gender.png")
plt.close()

# ---------------------------------------------------------------------------
# 6. Chart 3 — Survival by passenger class
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="Pclass", hue="Survived", palette=["#e74c3c", "#2ecc71"])
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("chart3_survival_by_class.png")
plt.close()

# ---------------------------------------------------------------------------
# 7. Chart 4 — Age distribution
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack",
             bins=20, palette=["#e74c3c", "#2ecc71"])
plt.title("Age Distribution of Passengers (by Survival)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("chart4_age_distribution.png")
plt.close()

# ---------------------------------------------------------------------------
# 8. Chart 5 — Fare by class
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Pclass", palette="viridis", legend=False)
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("chart5_fare_by_class.png")
plt.close()

# ---------------------------------------------------------------------------
# 9. Chart 6 — Correlation heatmap
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
numeric_df = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("chart6_correlation_heatmap.png")
plt.close()

print("\nAll 6 charts saved successfully.")
print("EDA complete.")
