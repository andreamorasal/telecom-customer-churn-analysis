
# 1. Project setup

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/myuser/datasets/telecom_customers_churn.csv")

print(df.shape)


# 2. Data profiling and quality

# Data quality report

data_quality_report = pd.DataFrame()

data_quality_report["data_type"] = df.dtypes
data_quality_report["null_count"] = df.isnull().sum()
data_quality_report["null_percentage"] = (df.isnull().mean() * 100).round(2)
data_quality_report["distinct_count"] = df.nunique()
data_quality_report["duplicate_count_rows"] = df.duplicated().sum()
data_quality_report["duplicate_value_count"] = df.apply(lambda x: x.duplicated().sum())

data_quality_report["min_value"] = df.select_dtypes(include="number").min()
data_quality_report["max_value"] = df.select_dtypes(include="number").max()

print(data_quality_report)


# Data type validation and correction

invalid_values = df["TotalCharges"][
    pd.to_numeric(df["TotalCharges"], errors="coerce").isna()
    & df["TotalCharges"].notna()
]

invalid_count = invalid_values.count()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Column categorisation

def categorize_columns(df):
    numerical = []
    categorical = []
    binary = []
    identifier = []

    for column in df.columns:
        if column.lower().endswith("id"):
            identifier.append(column)
        elif df[column].nunique() == 2:
            binary.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            numerical.append(column)
        else:
            categorical.append(column)

    return {
        "Numerical": numerical,
        "Categorical": categorical,
        "Binary": binary,
        "Identifier": identifier
    }


categories = categorize_columns(df)


# Total charges analysis

df["expected_total_charges"] = df["MonthlyCharges"] * df["tenure"]

df["difference_percentage"] = (
    abs(df["TotalCharges"] - df["expected_total_charges"])
    / df["expected_total_charges"].replace(0, np.nan)
) * 100

affected_customers = df[
    df["difference_percentage"] > 10
]

affected_customer_details = affected_customers[
    [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "expected_total_charges",
        "difference_percentage"
    ]
]


# Categorical analysis of affected customers

possible_causes = []

for column in categories["Categorical"] + categories["Binary"]:
    affected_pct = (
        affected_customers[column].value_counts(normalize=True) * 100
    )

    all_pct = df[column].value_counts(normalize=True) * 100

    comparison = pd.DataFrame({
        "Affected %": affected_pct,
        "All %": all_pct
    }).fillna(0)

    comparison["Difference"] = (
        comparison["Affected %"] - comparison["All %"]
    )

    comparison["Column"] = column
    comparison["Category"] = comparison.index

    possible_causes.append(comparison)

possible_causes = pd.concat(possible_causes, ignore_index=True)

overrepresented_categories = possible_causes.sort_values(
    "Difference",
    ascending=False
).head(10)


# Numerical comparison of affected and unaffected customers

possible_causes_numerical = df.groupby(
    df["difference_percentage"] > 10
)[["tenure", "MonthlyCharges", "TotalCharges"]].agg(
    ["mean", "median"]
)

print(possible_causes_numerical)


# 3. Customer churn analysis

# Churn rates by customer characteristics

churn_rate_gender = (
    df.groupby("gender")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

churn_rate_senior = (
    df.groupby("SeniorCitizen")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .rename(index={1: "Yes", 0: "No"})
)

churn_rate_partner = (
    df.groupby("Partner")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

churn_rate_dependents = (
    df.groupby("Dependents")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

ranking_higher_risk_customer_groups = pd.concat(
    [
        churn_rate_gender,
        churn_rate_senior,
        churn_rate_partner,
        churn_rate_dependents
    ],
    keys=["Gender", "Senior Citizen", "Partner", "Dependents"]
).sort_values(ascending=False)

print(ranking_higher_risk_customer_groups)


# Customer segmentation by tenure

def create_customer_segment_tenure(tenure):
    if tenure <= 12:
        return "New"
    elif tenure <= 24:
        return "Early"
    elif tenure <= 48:
        return "Established"
    else:
        return "Long-term"


df["customer_segment_tenure"] = df["tenure"].apply(
    create_customer_segment_tenure
)

churn_segment = (
    df.groupby("customer_segment_tenure")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)


# High churn customer profiles

top_10_customers_profiles_to_churn = (
    df.groupby(
        ["Contract", "PaymentMethod", "InternetService"]
    )["Churn"]
    .agg(
        customers="size",
        churned=lambda x: (x == "Yes").sum()
    )
    .reset_index()
)

top_10_customers_profiles_to_churn["churn_rate"] = (
    top_10_customers_profiles_to_churn["churned"]
    / top_10_customers_profiles_to_churn["customers"]
    * 100
)

top_10_customers_profiles_to_churn = (
    top_10_customers_profiles_to_churn
    .sort_values("churn_rate", ascending=False)
    .head(10)
)


# 4. Customer lifetime value analysis

df["CLV"] = df["MonthlyCharges"] * df["tenure"]

clv_churn_vs_retained = (
    df.groupby("Churn")
    .agg(
        customers=("CLV", "size"),
        average_clv=("CLV", "mean")
    )
    .reset_index()
)

clv_churn_vs_retained = clv_churn_vs_retained.round(2)


# 5. Customer risk analysis

# Churn by service and contract

churn_rate_online_security = (
    df.groupby("OnlineSecurity")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

churn_rate_tech_support = (
    df.groupby("TechSupport")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

churn_rate_contract = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)


# Customer risk score

df["customer_risk_score"] = (
    (df["Contract"] == "Month-to-month").astype(int)
    + (df["tenure"] < 12).astype(int)
    + (df["TechSupport"] == "No").astype(int)
    + (df["OnlineSecurity"] == "No").astype(int)
)

customer_risk_analysis = df[
    [
        "customerID",
        "Contract",
        "tenure",
        "TechSupport",
        "OnlineSecurity",
        "customer_risk_score",
        "Churn"
    ]
].copy()

customer_risk_analysis = customer_risk_analysis.sort_values(
    "customer_risk_score",
    ascending=False
)


# 6. Create screenshots

screenshot_dir = Path("/Users/myuser/telecom-customer-churn-analysis/screenshots")

print("Screenshots folder:", screenshot_dir.resolve())


# 1. Data quality report

fig, ax = plt.subplots(figsize=(14, 8))

report = data_quality_report[
    [
        "data_type",
        "null_count",
        "null_percentage",
        "distinct_count",
        "min_value",
        "max_value"
    ]
].astype(str)

ax.axis("off")
ax.set_title("Data Quality Report", pad=20)

table = ax.table(
    cellText=report.values,
    colLabels=report.columns,
    rowLabels=report.index,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)

plt.savefig(
    screenshot_dir / "data_quality_report.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# 2. Customer churn analysis

churn_rates = pd.concat(
    [
        churn_rate_gender.rename_axis("Group"),
        churn_rate_senior.rename_axis("Group"),
        churn_rate_partner.rename_axis("Group"),
        churn_rate_dependents.rename_axis("Group")
    ],
    keys=[
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents"
    ],
    names=["Characteristic", "Group"]
).rename("Churn Rate").reset_index()

churn_rates["Label"] = (
    churn_rates["Characteristic"]
    + ": "
    + churn_rates["Group"].astype(str)
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    churn_rates["Label"],
    churn_rates["Churn Rate"]
)

ax.set_xlabel("Churn Rate (%)")
ax.set_title("Customer Churn Rates by Customer Characteristics")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(
    screenshot_dir / "customer_churn_analysis.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# 3. Customer segmentation and churn

segment_order = ["New", "Early", "Established", "Long-term"]
segment_rates = churn_segment.reindex(segment_order)

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(segment_rates.index, segment_rates.values)
ax.set_xlabel("Customer Tenure Segment")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Customer Tenure")
ax.set_ylim(0, max(segment_rates.max() + 10, 100))

for i, value in enumerate(segment_rates.values):
    if pd.notna(value):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.tight_layout()
plt.savefig(
    screenshot_dir / "customer_segmentation_churn.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# 4. Customer risk analysis

risk_distribution = (
    customer_risk_analysis["customer_risk_score"]
    .value_counts()
    .reindex(range(5), fill_value=0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    risk_distribution.index.astype(str),
    risk_distribution.values
)

ax.set_xlabel("Customer Risk Score (0–4)")
ax.set_ylabel("Number of Customers")
ax.set_title("Distribution of Customer Risk Scores")

plt.tight_layout()
plt.savefig(
    screenshot_dir / "customer_risk_analysis.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# 5. Customer lifetime value

fig, ax = plt.subplots(figsize=(7, 5))

clv_data = clv_churn_vs_retained.copy()

ax.bar(
    clv_data["Churn"].map({
        "No": "Retained",
        "Yes": "Churned"
    }),
    clv_data["average_clv"]
)

ax.set_xlabel("Customer Status")
ax.set_ylabel("Average Customer Lifetime Value")
ax.set_title("Average CLV: Retained vs Churned Customers")

for i, value in enumerate(clv_data["average_clv"]):
    ax.text(i, value + 30, f"{value:,.0f}", ha="center")

plt.tight_layout()
plt.savefig(
    screenshot_dir / "customer_lifetime_value.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# Check saved screenshots

print("Screenshots saved:")

for file in sorted(screenshot_dir.glob("*.png")):
    print(f"✓ {file.name}")

print(f"\nTotal screenshots: {len(list(screenshot_dir.glob('*.png')))}")
print(f"Folder location: {screenshot_dir.resolve()}")
