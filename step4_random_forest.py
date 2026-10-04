# ============================================================
# STEP 4: RANDOM FOREST CLASSIFIER
# Crowdfunding Campaign Performance Predictor
# EcoSip Smart Bottle
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. LOAD PROCESSED DATASET
# ------------------------------------------------------------

input_file = "crowdfunding_processed.csv"

df = pd.read_csv(input_file)

print("\n========================================")
print("STEP 4: RANDOM FOREST CLASSIFIER")
print("========================================")

print("\nDataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. DEFINE TARGET VARIABLE
# ------------------------------------------------------------

target = "success"

# Remove rows where target is missing
df = df.dropna(subset=[target])

df[target] = df[target].astype(int)


# ------------------------------------------------------------
# 3. SELECT PRE-LAUNCH FEATURES
# ------------------------------------------------------------
# IMPORTANT:
# We deliberately DO NOT use final campaign outcomes such as:
#
# amount_raised
# funding_percentage
# number_of_backers
# average_contribution
# campaign_status
# first_24h_funding
# first_7d_funding
# conversion_rate
#
# These would create data leakage for a pre-launch prediction model.

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

# Keep only columns that actually exist
categorical_features = [
    col for col in categorical_features
    if col in df.columns
]

numerical_features = [
    col for col in numerical_features
    if col in df.columns
]

selected_features = categorical_features + numerical_features

X = df[selected_features]
y = df[target]


# ------------------------------------------------------------
# 4. DISPLAY FEATURES
# ------------------------------------------------------------

print("\nFeatures used for prediction:")

for feature in selected_features:
    print("-", feature)

print("\nTarget variable:")
print("success")


# ------------------------------------------------------------
# 5. CHECK TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n========================================")
print("TARGET DISTRIBUTION")
print("========================================")

print(y.value_counts())

print("\nTarget percentages:")
print(
    y.value_counts(normalize=True).mul(100).round(2)
)


# ------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 7. PREPROCESSING
# ------------------------------------------------------------

# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)


# ------------------------------------------------------------
# 8. RANDOM FOREST MODEL
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced"
)


# ------------------------------------------------------------
# 9. COMPLETE MODEL PIPELINE
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            random_forest
        )
    ]
)


# ------------------------------------------------------------
# 10. TRAIN MODEL
# ------------------------------------------------------------

print("\n========================================")
print("TRAINING RANDOM FOREST MODEL")
print("========================================")

model.fit(X_train, y_train)

print("Model training completed successfully.")


# ------------------------------------------------------------
# 11. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 12. MODEL EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ------------------------------------------------------------
# 13. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Unsuccessful",
            "Successful"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 14. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# Display confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Unsuccessful",
        "Successful"
    ]
).plot()

plt.title(
    "Random Forest - Crowdfunding Success Prediction"
)

plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 15. FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

# Get transformed feature names
feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get Random Forest importance
importance_values = (
    model
    .named_steps["classifier"]
    .feature_importances_
)

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance_values
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 important features:")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 16. SAVE FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)


# ------------------------------------------------------------
# 17. SAVE MODEL
# ------------------------------------------------------------

model_file = "crowdfunding_random_forest_model.pkl"

joblib.dump(
    model,
    model_file
)

print("\nModel saved as:")
print(model_file)


# ------------------------------------------------------------
# 18. SAVE MODEL RESULTS
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    }
)

results.to_csv(
    "random_forest_model_results.csv",
    index=False
)


# ------------------------------------------------------------
# 19. FINAL MESSAGE
# ------------------------------------------------------------

print("\n========================================")
print("STEP 4 COMPLETED")
print("========================================")

print("\nFiles created:")

print("1. crowdfunding_random_forest_model.pkl")
print("2. random_forest_model_results.csv")
print("3. random_forest_feature_importance.csv")
print("4. random_forest_confusion_matrix.png")

print("\nRandom Forest training and evaluation completed.")