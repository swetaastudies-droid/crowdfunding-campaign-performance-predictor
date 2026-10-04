# ============================================================
# STEP 3: DATA PROCESSING & CLEANING
# Crowdfunding Campaign Performance Predictor
# EcoSip Smart Bottle
# ============================================================

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 1. LOAD THE HISTORICAL DATASET
# ------------------------------------------------------------

input_file = "crowdfunding_historical_improved_500.csv"

df = pd.read_csv(input_file)

print("\n========================================")
print("STEP 3: DATA PROCESSING & CLEANING")
print("========================================")

print("\nOriginal dataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n========================================")
print("MISSING VALUES BEFORE CLEANING")
print("========================================")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])


# ------------------------------------------------------------
# 3. REMOVE DUPLICATE CAMPAIGNS
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()

print("Dataset shape after removing duplicates:")
print(df.shape)


# ------------------------------------------------------------
# 4. CLEAN TEXT COLUMNS
# ------------------------------------------------------------

# Description is required for sentiment analysis.
# Missing descriptions are replaced with an empty string.

if "description" in df.columns:
    df["description"] = df["description"].fillna("").astype(str)

# Clean category-related columns
for column in ["category", "subcategory", "country", "launch_month"]:
    if column in df.columns:
        df[column] = df[column].fillna("Unknown").astype(str)


# ------------------------------------------------------------
# 5. CLEAN NUMERICAL COLUMNS
# ------------------------------------------------------------

numeric_columns = [
    "funding_goal",
    "amount_raised",
    "funding_percentage",
    "number_of_backers",
    "average_contribution",
    "campaign_duration",
    "reward_tiers",
    "lowest_reward",
    "average_reward",
    "highest_reward",
    "number_of_images",
    "story_length",
    "social_media_followers",
    "email_list_size",
    "pre_launch_community_size",
    "marketing_spend",
    "landing_page_traffic",
    "campaign_page_visits",
    "conversion_rate",
    "updates_posted",
    "comments_engagement",
    "first_24h_funding",
    "first_7d_funding"
]

for column in numeric_columns:

    if column in df.columns:

        # Convert values to numeric
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Replace missing values with median
        df[column] = df[column].fillna(
            df[column].median()
        )


# ------------------------------------------------------------
# 6. CLEAN BOOLEAN COLUMNS
# ------------------------------------------------------------

if "video_available" in df.columns:

    df["video_available"] = (
        df["video_available"]
        .astype(str)
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })
    )

    df["video_available"] = df["video_available"].fillna(0)


# ------------------------------------------------------------
# 7. SENTIMENT ANALYSIS
# ------------------------------------------------------------

print("\n========================================")
print("SENTIMENT ANALYSIS")
print("========================================")

def calculate_sentiment(text):

    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0


if "description" in df.columns:

    df["sentiment_polarity"] = df["description"].apply(
        calculate_sentiment
    )

    print("\nSentiment analysis completed.")

    print("\nSample sentiment results:")

    print(
        df[
            [
                "description",
                "sentiment_polarity"
            ]
        ].head(10)
    )

    # --------------------------------------------------------
    # Create sentiment categories
    # --------------------------------------------------------

    def sentiment_category(score):

        if score > 0.05:
            return "Positive"

        elif score < -0.05:
            return "Negative"

        else:
            return "Neutral"

    df["sentiment_category"] = df[
        "sentiment_polarity"
    ].apply(sentiment_category)

    print("\nSentiment distribution:")

    print(
        df["sentiment_category"].value_counts()
    )


# ------------------------------------------------------------
# 8. ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------

print("\n========================================")
print("CATEGORY ENCODING")
print("========================================")

categorical_columns = [
    "category",
    "subcategory",
    "country",
    "launch_month"
]

# Label encoding for analysis/model preparation
label_encoders = {}

for column in categorical_columns:

    if column in df.columns:

        encoder = LabelEncoder()

        df[column + "_encoded"] = encoder.fit_transform(
            df[column].astype(str)
        )

        label_encoders[column] = encoder

        print(
            column,
            "encoded successfully."
        )


# ------------------------------------------------------------
# 9. CREATE SUCCESS LABEL
# ------------------------------------------------------------

if "success" in df.columns:

    df["success"] = pd.to_numeric(
        df["success"],
        errors="coerce"
    )

    df["success"] = df["success"].fillna(0).astype(int)

    print("\nSuccess distribution:")

    print(
        df["success"].value_counts()
    )


# ------------------------------------------------------------
# 10. CHECK FINAL MISSING VALUES
# ------------------------------------------------------------

print("\n========================================")
print("MISSING VALUES AFTER CLEANING")
print("========================================")

remaining_missing = df.isnull().sum()

print(
    remaining_missing[
        remaining_missing > 0
    ]
)


# ------------------------------------------------------------
# 11. SAVE PROCESSED DATASET
# ------------------------------------------------------------

output_file = "crowdfunding_processed.csv"

df.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========================================")
print("STEP 3 COMPLETED")
print("========================================")

print("\nFinal dataset shape:")
print(df.shape)

print("\nProcessed dataset saved as:")
print(output_file)

print("\n========================================")
print("PROCESSING COMPLETE")
print("========================================")