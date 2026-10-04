# ============================================================
# STEP 2: CROWDFUNDING DATA COLLECTION
# Load CSV and extract important features
# ============================================================

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "crowdfunding_historical_synthetic_500.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print()


# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())


# ------------------------------------------------------------
# 3. DISPLAY ALL COLUMN NAMES
# ------------------------------------------------------------

print("\nAvailable columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

print("\nStandardized columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 5. EXTRACT FUNDING GOAL
# ------------------------------------------------------------

# Different crowdfunding datasets may use different
# column names for funding goal.

goal_columns = [
    "goal",
    "funding_goal",
    "usd_goal_real",
    "usd_goal"
]

goal_column = None

for column in goal_columns:

    if column in df.columns:
        goal_column = column
        break


if goal_column:

    df["funding_goal"] = pd.to_numeric(
        df[goal_column],
        errors="coerce"
    )

    print(
        "\nFunding goal extracted from:",
        goal_column
    )

else:

    print(
        "\nWARNING: Funding goal column not found."
    )


# ------------------------------------------------------------
# 6. EXTRACT CATEGORY
# ------------------------------------------------------------

category_columns = [
    "category",
    "main_category",
    "subcategory"
]

category_column = None

for column in category_columns:

    if column in df.columns:
        category_column = column
        break


if category_column:

    df["category"] = (
        df[category_column]
        .astype(str)
        .str.strip()
    )

    print(
        "Category extracted from:",
        category_column
    )

else:

    print(
        "WARNING: Category column not found."
    )


# ------------------------------------------------------------
# 7. CALCULATE CAMPAIGN DURATION
# ------------------------------------------------------------

# Kickstarter datasets commonly contain:
# deadline and launched

if (
    "deadline" in df.columns
    and "launched" in df.columns
):

    df["deadline"] = pd.to_datetime(
        df["deadline"],
        errors="coerce"
    )

    df["launched"] = pd.to_datetime(
        df["launched"],
        errors="coerce"
    )

    df["duration"] = (
        df["deadline"] - df["launched"]
    ).dt.days

    print(
        "Campaign duration calculated "
        "from deadline and launched."
    )

elif "duration" in df.columns:

    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="coerce"
    )

    print(
        "Campaign duration extracted "
        "from existing duration column."
    )

else:

    print(
        "WARNING: Duration information not found."
    )


# ------------------------------------------------------------
# 8. CREATE SUCCESS / FAILURE VARIABLE
# ------------------------------------------------------------

# Kickstarter commonly uses the "state" column.

if "state" in df.columns:

    df["success"] = np.where(
        df["state"]
        .astype(str)
        .str.lower()
        .eq("successful"),
        1,
        0
    )

    print(
        "Success/failure extracted from state."
    )


elif "status" in df.columns:

    df["success"] = np.where(
        df["status"]
        .astype(str)
        .str.lower()
        .isin(["successful", "success"]),
        1,
        0
    )

    print(
        "Success/failure extracted from status."
    )


elif "success" in df.columns:

    df["success"] = (
        df["success"]
        .astype(str)
        .str.lower()
        .map({
            "successful": 1,
            "success": 1,
            "yes": 1,
            "true": 1,
            "1": 1,
            "failed": 0,
            "failure": 0,
            "no": 0,
            "false": 0,
            "0": 0
        })
    )

    print(
        "Existing success column converted."
    )

else:

    print(
        "WARNING: Success/failure column not found."
    )


# ------------------------------------------------------------
# 9. CREATE FINAL FEATURE DATASET
# ------------------------------------------------------------

important_features = [
    "funding_goal",
    "category",
    "duration",
    "success"
]

available_features = [
    column
    for column in important_features
    if column in df.columns
]

campaign_features = df[
    available_features
].copy()


# ------------------------------------------------------------
# 10. REMOVE MISSING VALUES
# ------------------------------------------------------------

campaign_features = campaign_features.dropna(
    subset=available_features
)


# ------------------------------------------------------------
# 11. REMOVE INVALID DURATION VALUES
# ------------------------------------------------------------

if "duration" in campaign_features.columns:

    campaign_features = campaign_features[
        campaign_features["duration"] > 0
    ]


# ------------------------------------------------------------
# 12. DISPLAY EXTRACTED FEATURES
# ------------------------------------------------------------

print("\n==========================================")
print("IMPORTANT CAMPAIGN FEATURES")
print("==========================================")

print(
    campaign_features.head(10)
)


# ------------------------------------------------------------
# 13. DATASET SUMMARY
# ------------------------------------------------------------

print("\nExtracted dataset shape:")
print(
    campaign_features.shape
)


# ------------------------------------------------------------
# 14. SUCCESS / FAILURE SUMMARY
# ------------------------------------------------------------

if "success" in campaign_features.columns:

    print("\nCampaign outcome distribution:")

    print(
        campaign_features["success"]
        .value_counts()
        .rename({
            0: "Unsuccessful",
            1: "Successful"
        })
    )


# ------------------------------------------------------------
# 15. SAVE EXTRACTED DATASET
# ------------------------------------------------------------

output_file = "crowdfunding_important_features.csv"

campaign_features.to_csv(
    output_file,
    index=False
)

print("\n==========================================")
print("STEP 2 COMPLETED")
print("==========================================")

print(
    "Extracted dataset saved as:",
    output_file
)