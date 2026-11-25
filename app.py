# =====================================================
# PROJECT: CUSTOMER CHURN ANALYSIS
# TASK 2 - DATA ANALYTICS
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization settings
sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("             CUSTOMER CHURN ANALYSIS")
print("=" * 60)

# -----------------------------------------------------
# STEP 1: LOAD DATASET
# -----------------------------------------------------

df = pd.read_csv("Telco-Customer-Churn.csv")

print("\nSTEP 1: DATASET LOADED SUCCESSFULLY")
print("Number of rows and columns:", df.shape)

# -----------------------------------------------------
# STEP 2: UNDERSTAND THE DATA
# -----------------------------------------------------

print("\nSTEP 2: FIRST FIVE RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all"))

# -----------------------------------------------------
# STEP 3: DATA CLEANING
# -----------------------------------------------------

print("\nSTEP 3: DATA CLEANING")

# Remove duplicate records
print("Duplicate records:", df.duplicated().sum())
df = df.drop_duplicates()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing numerical values with median
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# Fill missing categorical values with mode
categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("Data cleaning completed successfully!")

# Save cleaned dataset
df.to_csv("outputs/cleaned_customer_data.csv", index=False)

# -----------------------------------------------------
# STEP 4: CUSTOMER CHURN ANALYSIS
# -----------------------------------------------------

print("\nSTEP 4: CUSTOMER CHURN ANALYSIS")

churn_counts = df["Churn"].value_counts()

print("\nCustomer churn count:")
print(churn_counts)

churn_percentage = (
    df["Churn"].value_counts(normalize=True) * 100
)

print("\nCustomer churn percentage:")
print(churn_percentage.round(2))

total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
retained_customers = (df["Churn"] == "No").sum()

churn_rate = (
    churned_customers / total_customers
) * 100

print("\nTotal customers:", total_customers)
print("Customers who left:", churned_customers)
print("Customers retained:", retained_customers)
print("Overall churn rate:", round(churn_rate, 2), "%")

# -----------------------------------------------------
# STEP 5: GRAPH 1 - CHURN DISTRIBUTION
# -----------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/1_churn_distribution.png")
plt.close()

# -----------------------------------------------------
# STEP 6: GRAPH 2 - CONTRACT ANALYSIS
# -----------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn",
    palette="Set1"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/2_contract_analysis.png")
plt.close()

# -----------------------------------------------------
# STEP 7: GRAPH 3 - MONTHLY CHARGES
# -----------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("outputs/3_monthly_charges.png")
plt.close()

# -----------------------------------------------------
# STEP 8: GRAPH 4 - CUSTOMER TENURE
# -----------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    multiple="stack"
)

plt.title("Customer Tenure and Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/4_tenure_analysis.png")
plt.close()

# -----------------------------------------------------
# STEP 9: GRAPH 5 - PAYMENT METHOD
# -----------------------------------------------------

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn",
    palette="Set2"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/5_payment_analysis.png")
plt.close()

# -----------------------------------------------------
# STEP 10: GRAPH 6 - CORRELATION HEATMAP
# -----------------------------------------------------

df["ChurnNumeric"] = df["Churn"].map(
    {"No": 0, "Yes": 1}
)

correlation_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ChurnNumeric"
]

correlation = df[correlation_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Customer Churn Correlation Analysis")
plt.tight_layout()
plt.savefig("outputs/6_correlation_heatmap.png")
plt.close()

# -----------------------------------------------------
# STEP 11: CONTRACT-WISE CHURN RATE
# -----------------------------------------------------

print("\nSTEP 11: CHURN RATE BY CONTRACT")

contract_churn = df.groupby("Contract")[
    "ChurnNumeric"
].mean() * 100

print(contract_churn.round(2))

# -----------------------------------------------------
# STEP 12: AVERAGE CUSTOMER CHARGES
# -----------------------------------------------------

print("\nSTEP 12: AVERAGE MONTHLY CHARGES")

print(
    df.groupby("Churn")["MonthlyCharges"].mean()
)

# -----------------------------------------------------
# STEP 13: AVERAGE CUSTOMER TENURE
# -----------------------------------------------------

print("\nSTEP 13: AVERAGE TENURE")

print(
    df.groupby("Churn")["tenure"].mean()
)

# -----------------------------------------------------
# STEP 14: KEY FINDINGS
# -----------------------------------------------------

print("\n" + "=" * 60)
print("                    KEY FINDINGS")
print("=" * 60)

print("1. Total customers:", total_customers)

print("2. Customers who cancelled:",
      churned_customers)

print("3. Customers retained:",
      retained_customers)

print("4. Customer churn rate:",
      round(churn_rate, 2), "%")

print("\n5. Contract-wise churn rate:")
print(contract_churn.round(2))

print("\n6. Average monthly charges by churn status:")
print(df.groupby("Churn")["MonthlyCharges"].mean())

print("\n7. Average tenure by churn status:")
print(df.groupby("Churn")["tenure"].mean())

# -----------------------------------------------------
# STEP 15: RECOMMENDATIONS
# -----------------------------------------------------

print("\n" + "=" * 60)
print("             BUSINESS RECOMMENDATIONS")
print("=" * 60)

print("""
1. Provide special discounts to customers with
   high monthly charges.

2. Encourage customers to choose long-term contracts.

3. Improve customer support and service quality.

4. Introduce loyalty rewards for existing customers.

5. Identify customers who may cancel their subscriptions
   and offer them suitable retention plans.

6. Make payment methods convenient and reliable.

7. Regularly monitor customer satisfaction.
""")

# -----------------------------------------------------
# STEP 16: SAVE FINAL RESULTS
# -----------------------------------------------------

df.drop(columns=["ChurnNumeric"]).to_csv(
    "outputs/cleaned_customer_data.csv",
    index=False
)

contract_churn.to_csv(
    "outputs/contract_churn_rates.csv"
)

summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Churned Customers",
        "Retained Customers",
        "Churn Rate Percentage"
    ],
    "Value": [
        total_customers,
        churned_customers,
        retained_customers,
        round(churn_rate, 2)
    ]
})

summary.to_csv("outputs/churn_summary.csv", index=False)

print("\n" + "=" * 60)
print("         PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nAll graphs and result files are saved")
print("inside the outputs folder.")