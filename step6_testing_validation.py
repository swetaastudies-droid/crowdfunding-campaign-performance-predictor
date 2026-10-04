# ============================================================
# STEP 6: TESTING & VALIDATION
# Crowdfunding Campaign Performance Predictor
# EcoSip
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("\n========================================")
print("STEP 6: TESTING & VALIDATION")
print("========================================")


# ------------------------------------------------------------
# 1. LOAD TRAINING DATA AND MODEL
# ------------------------------------------------------------

training_file = "crowdfunding_processed.csv"

model_file = "improved_crowdfunding_random_forest.pkl"

training_data = pd.read_csv(training_file)

model = joblib.load(model_file)

print("\nTraining dataset loaded.")
print("Trained model loaded.")


# ------------------------------------------------------------
# 2. RECENT REAL CAMPAIGNS
# ------------------------------------------------------------
# Actual campaign information is based on public campaign pages.
#
# Funding goals are converted to INR only to make the monetary
# input consistent with our EcoSip model.
#
# Exchange rates used for standardisation:
# USD = 96.15 INR
# EUR = 108.395 INR
# GBP = 127.515 INR
#
# These are current indicative rates, NOT historical transaction
# rates. They are used only to standardise model input.


exchange_rates = {
    "USD": 96.15,
    "EUR": 108.395,
    "GBP": 127.515
}


real_campaigns = [

    {
        "campaign_name":
        "iUVì – Smart Water Bottle with UV-C Purification",

        "category": "Technology",
        "subcategory": "Smart Devices",
        "country": "Italy",
        "launch_month": "May",

        "goal_original": 100,
        "currency": "EUR",

        "amount_raised_original": 5846,
        "backers": 87,
        "duration": 33,

        "actual_success": 1
    },


    {
        "campaign_name":
        "MERIDIAN - The Titanium Cup That Does It All",

        "category": "Consumer Products",
        "subcategory": "Lifestyle",
        "country": "Germany",
        "launch_month": "October",

        "goal_original": 2000,
        "currency": "EUR",

        "amount_raised_original": 5025,
        "backers": 111,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "DABEI Travelpods - Travel Containers",

        "category": "Consumer Products",
        "subcategory": "Home",
        "country": "Germany",
        "launch_month": "March",

        "goal_original": 3000,
        "currency": "EUR",

        "amount_raised_original": 20042,
        "backers": 244,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "Magnetic 12-in-1 Phone Stand & Titanium EDC Tool",

        "category": "Consumer Products",
        "subcategory": "Lifestyle",
        "country": "United States",
        "launch_month": "June",

        "goal_original": 2000,
        "currency": "USD",

        "amount_raised_original": 13451,
        "backers": 159,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "MIDISHI M1 Automatic Shoe Washer & Dryer",

        "category": "Consumer Products",
        "subcategory": "Home",
        "country": "Canada",
        "launch_month": "April",

        "goal_original": 2000,
        "currency": "USD",

        "amount_raised_original": 195716,
        "backers": 519,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "NanoB10 - 9-Mode EDC Flashlight",

        "category": "Technology",
        "subcategory": "Electronics",
        "country": "United Kingdom",
        "launch_month": "January",

        "goal_original": 3000,
        "currency": "GBP",

        "amount_raised_original": 55165,
        "backers": 573,
        "duration": 42,

        "actual_success": 1
    },


    {
        "campaign_name":
        "VEXTAKI - 9-in-1 Multi-Tool EDC Pen",

        "category": "Consumer Products",
        "subcategory": "Lifestyle",
        "country": "United States",
        "launch_month": "May",

        "goal_original": 2000,
        "currency": "USD",

        "amount_raised_original": 2899,
        "backers": 65,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "GYMBLY Home Reformer",

        "category": "Consumer Products",
        "subcategory": "Lifestyle",
        "country": "United States",
        "launch_month": "October",

        "goal_original": 2000,
        "currency": "USD",

        "amount_raised_original": 100110,
        "backers": 305,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "Magnetic Switch 2 Sling",

        "category": "Consumer Products",
        "subcategory": "Lifestyle",
        "country": "United States",
        "launch_month": "October",

        "goal_original": 2000,
        "currency": "USD",

        "amount_raised_original": 5155,
        "backers": 60,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "PickTwist 3D Guitar Pick",

        "category": "Consumer Products",
        "subcategory": "Lifestyle",
        "country": "Spain",
        "launch_month": "February",

        "goal_original": 2900,
        "currency": "EUR",

        "amount_raised_original": 12844,
        "backers": 657,
        "duration": 30,

        "actual_success": 1
    },


    {
        "campaign_name":
        "Piper Initial Campaign - Cancelled",

        "category": "Consumer Products",
        "subcategory": "Lifestyle",
        "country": "Singapore",
        "launch_month": "April",

        "goal_original": 40000,
        "currency": "USD",

        "amount_raised_original": 15600,
        "backers": 0,
        "duration": 14,

        "actual_success": 0
    }
]


# ------------------------------------------------------------
# 3. CREATE VALIDATION DATAFRAME
# ------------------------------------------------------------

validation = pd.DataFrame(
    real_campaigns
)


# ------------------------------------------------------------
# 4. CONVERT FUNDING GOAL TO INR
# ------------------------------------------------------------

validation["funding_goal"] = (
    validation["goal_original"] *
    validation["currency"].map(exchange_rates)
)


validation["funding_goal"] = (
    validation["funding_goal"].round(0)
)


# ------------------------------------------------------------
# 5. DISPLAY ACTUAL FUNDING PERFORMANCE
# ------------------------------------------------------------

validation["actual_funding_percentage"] = (
    validation["amount_raised_original"] /
    validation["goal_original"] *
    100
)


validation["actual_funding_percentage"] = (
    validation["actual_funding_percentage"]
    .round(2)
)


# ------------------------------------------------------------
# 6. PREPARE MODEL INPUTS
# ------------------------------------------------------------

categorical_features = [
    "category",
    "subcategory",
    "country",
    "launch_month"
]

numerical_features = [
    "funding_goal",
    "campaign_duration",
    "reward_tiers",
    "lowest_reward",
    "average_reward",
    "highest_reward",
    "video_available",
    "number_of_images",
    "story_length",
    "social_media_followers",
    "email_list_size",
    "pre_launch_community_size",
    "marketing_spend",
    "landing_page_traffic",
    "sentiment_polarity"
]


# ------------------------------------------------------------
# 7. USE TRAINING DATA MEDIANS FOR UNAVAILABLE PUBLIC DATA
# ------------------------------------------------------------
# IMPORTANT:
#
# Public campaign pages do NOT consistently disclose:
# - email list
# - marketing spend
# - pre-launch community
# - landing-page traffic
# - story word count
# - reward structure
# - social followers at launch
#
# We therefore use the training-data median/mode rather than
# inventing values.
#
# This makes the test an EXTERNAL SANITY CHECK rather than a
# definitive real-world validation study.


for column in categorical_features:

    if column not in validation.columns:
        validation[column] = (
            training_data[column].mode()[0]
        )


for column in numerical_features:

    if column == "funding_goal":
        continue

    if column == "campaign_duration":

        continue

    if column in training_data.columns:

        validation[column] = (
            training_data[column].median()
        )


# ------------------------------------------------------------
# 8. USE OBSERVED CAMPAIGN DURATION
# ------------------------------------------------------------

validation["campaign_duration"] = (
    validation["duration"]
)


# ------------------------------------------------------------
# 9. SET VIDEO AVAILABILITY
# ------------------------------------------------------------
# Campaign pages used in this sample display campaign video
# thumbnails. We use 1 as an observed public-page feature.


validation["video_available"] = 1


# ------------------------------------------------------------
# 10. RUN MODEL
# ------------------------------------------------------------

model_features = (
    categorical_features +
    numerical_features
)


X_validation = validation[
    model_features
]


predicted = model.predict(
    X_validation
)


probabilities = model.predict_proba(
    X_validation
)[:, 1]


validation["predicted_success"] = (
    predicted
)


validation["success_probability"] = (
    probabilities * 100
).round(2)


# ------------------------------------------------------------
# 11. CONVERT PREDICTION TO TEXT
# ------------------------------------------------------------

validation["predicted_outcome"] = (
    validation["predicted_success"]
    .map({
        0: "Unsuccessful",
        1: "Successful"
    })
)


validation["actual_outcome"] = (
    validation["actual_success"]
    .map({
        0: "Unsuccessful",
        1: "Successful"
    })
)


# ------------------------------------------------------------
# 12. COMPARE PREDICTED VS ACTUAL
# ------------------------------------------------------------

validation["prediction_correct"] = (
    validation["predicted_success"] ==
    validation["actual_success"]
)


# ------------------------------------------------------------
# 13. DISPLAY VALIDATION RESULTS
# ------------------------------------------------------------

print("\n========================================")
print("PREDICTED VS ACTUAL RESULTS")
print("========================================")


display_columns = [
    "campaign_name",
    "actual_outcome",
    "predicted_outcome",
    "success_probability",
    "actual_funding_percentage",
    "prediction_correct"
]


print(
    validation[
        display_columns
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 14. CALCULATE METRICS
# ------------------------------------------------------------

y_true = validation[
    "actual_success"
]

y_pred = validation[
    "predicted_success"
]


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 15. DISPLAY VALIDATION METRICS
# ------------------------------------------------------------

print("\n========================================")
print("EXTERNAL VALIDATION RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ------------------------------------------------------------
# 16. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n========================================")
print("VALIDATION CONFUSION MATRIX")
print("========================================")

print(cm)


# ------------------------------------------------------------
# 17. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n========================================")
print("VALIDATION CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Unsuccessful",
            "Successful"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 18. SAVE VALIDATION RESULTS
# ------------------------------------------------------------

output_file = (
    "step6_real_campaign_validation_results.csv"
)


validation.to_csv(
    output_file,
    index=False
)


print("\n========================================")
print("STEP 6 COMPLETED")
print("========================================")

print(
    "\nValidation results saved as:"
)

print(output_file)

print(
    "\nImportant:"
)

print(
    "This is an external sanity check because "
    "several real-campaign pre-launch variables "
    "were not publicly available and were imputed "
    "from the synthetic training dataset."
)