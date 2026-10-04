# ============================================================
# STEP 4B: IMPROVED RANDOM FOREST CLASSIFIER
# Crowdfunding Campaign Performance Predictor
# EcoSip Smart Bottle
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
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
# 1. LOAD DATA
# ------------------------------------------------------------

input_file = "crowdfunding_processed.csv"

df = pd.read_csv(input_file)

print("\n========================================")
print("STEP 4B: IMPROVED RANDOM FOREST")
print("========================================")

print("\nDataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

target = "success"

df = df.dropna(subset=[target])

df[target] = df[target].astype(int)


# ------------------------------------------------------------
# 3. PRE-LAUNCH FEATURES
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

categorical_features = [
    c for c in categorical_features
    if c in df.columns
]

numerical_features = [
    c for c in numerical_features
    if c in df.columns
]

selected_features = (
    categorical_features +
    numerical_features
)

X = df[selected_features]
y = df[target]


# ------------------------------------------------------------
# 4. CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print(y.value_counts())

print("\nPercentage distribution:")

print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
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
# 6. PREPROCESSING
# ------------------------------------------------------------

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

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

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
# 7. IMPROVED RANDOM FOREST
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# 8. MODEL PIPELINE
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
# 9. TRAIN
# ------------------------------------------------------------

print("\n========================================")
print("TRAINING IMPROVED MODEL")
print("========================================")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ------------------------------------------------------------
# 10. PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 11. MODEL METRICS
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


# ------------------------------------------------------------
# 12. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n========================================")
print("IMPROVED MODEL PERFORMANCE")
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

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Unsuccessful",
        "Successful"
    ]
).plot()

plt.title(
    "Improved Random Forest - Crowdfunding Success"
)

plt.tight_layout()

plt.savefig(
    "improved_random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 15. CROSS-VALIDATION
# ------------------------------------------------------------

print("\n========================================")
print("5-FOLD CROSS-VALIDATION")
print("========================================")

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print(
    "F1 scores:",
    np.round(cv_scores, 4)
)

print(
    "Average CV F1:",
    round(cv_scores.mean(), 4)
)


# ------------------------------------------------------------
# 16. FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

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

print("\nTop 15 features:")

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)

feature_importance.to_csv(
    "improved_random_forest_feature_importance.csv",
    index=False
)


# ------------------------------------------------------------
# 17. SAVE MODEL RESULTS
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Average CV F1"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            cv_scores.mean()
        ]
    }
)

results.to_csv(
    "improved_random_forest_results.csv",
    index=False
)


# ------------------------------------------------------------
# 18. SAVE IMPROVED MODEL
# ------------------------------------------------------------

joblib.dump(
    model,
    "improved_crowdfunding_random_forest.pkl"
)


# ------------------------------------------------------------
# 19. FINAL MESSAGE
# ------------------------------------------------------------

print("\n========================================")
print("STEP 4B COMPLETED")
print("========================================")

print("\nFiles created:")

print("1. improved_crowdfunding_random_forest.pkl")
print("2. improved_random_forest_results.csv")
print("3. improved_random_forest_feature_importance.csv")
print("4. improved_random_forest_confusion_matrix.png")

print("\nImproved Random Forest analysis completed.")