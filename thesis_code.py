"""
Predicting Return Home Intentions of Syrian Refugees in Europe
After the Fall of Assad's Regime Using Machine Learning

Student     : Bayram Altintaş (u172831)
Supervisor  : Dr. Çiçek Güven
Institution : Tilburg University — Data Science & Society

Description
-----------
This script implements the full machine learning pipeline for the thesis.
It loads the UNHCR Intentions Survey 2025, preprocesses the data,
trains four classification models (Logistic Regression, Random Forest,
and Gradient Boosting), evaluates them using 5-fold stratified
cross-validation and a held-out test set, and produces all figures used
in the thesis.


Reproducibility
---------------
All random operations use RANDOM_SEED = 42. Results are fully
reproducible given the same dataset and library versions listed
in requirements.txt.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — saves figures to disk
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     RandomizedSearchCV, cross_validate)
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint, uniform, loguniform


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH   = "hh_anonym.xlsx"  # path to the UNHCR dataset
OUTPUT_DIR  = "outputs"         # folder where all figures are saved
RANDOM_SEED = 42                # single seed used for all random operations

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 — LOAD DATA#
# =============================================================================
df = pd.read_excel(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Map the full intention strings to short labels for easier handling.
outcome_map = {
    "I intend to stay":                             "Stay",
    "I am unsure of my future plans":               "Unsure",
    "I intend to move and live in a third country": "Move",
    "I intend to return permanently to Syria":      "Return",
}
df["y"] = df["future_intentions"].map(outcome_map)

# CLASS_NAMES is in alphabetical order because LabelEncoder sorts alphabetically.
# Move=0, Return=1, Stay=2, Unsure=3
CLASS_NAMES = ["Move", "Return", "Stay", "Unsure"]

print(f"\nClass distribution:")
for cls, n in zip(CLASS_NAMES,
                  [sum(df["y"] == c) for c in CLASS_NAMES]):
    print(f"  {cls:8s}: {n:5d}  ({n/len(df)*100:.1f}%)")


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING
# =============================================================================
df["arrival_year_num"]    = df["arrival_year"].replace({"Before 2011": 2010}).astype(int)
df["years_since_arrival"] = 2025 - df["arrival_year_num"]
df["family_size_num"]     = df["family_size"].replace("10+", 10).astype(float)


# =============================================================================
# SECTION 3 — MISSING VALUE HANDLING
# =============================================================================
for col in ["living_family_spouse", "living_family_child", "living_family_old"]:
    df[col] = df[col].fillna("Not reported")

df["current_country"] = df["current_country"].fillna("Others")

for col in ["gender", "age", "valid_residency", "main_activity"]:
    df[col] = df[col].fillna("Unknown")


# =============================================================================
# SECTION 4 — DEFINE FEATURES AND ENCODE OUTCOME
# =============================================================================
CAT_FEATURES = [
    "age", "gender", "legal_status", "valid_residency",
    "current_country", "living_family_spouse", "living_family_child",
    "living_family_old", "needs_care", "temp_visit",
    "governorate", "poses_document", "main_activity",
]
NUM_FEATURES = ["years_since_arrival", "family_size_num"]

X = df[CAT_FEATURES + NUM_FEATURES].copy()

le = LabelEncoder()
le.fit(CLASS_NAMES)
y = le.transform(df["y"].values)


# =============================================================================
# SECTION 5 — PREPROCESSING PIPELINE
# I use scikit-learn's Pipeline and ColumnTransformer to apply different
# preprocessing steps to categorical and numeric columns separately.
# =============================================================================
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipe, CAT_FEATURES),
    ("num", num_pipe, NUM_FEATURES),
])


# =============================================================================
# SECTION 6 — CLASS WEIGHTS
# =============================================================================
classes = np.arange(len(CLASS_NAMES))
cw      = compute_class_weight("balanced", classes=classes, y=y)
cw_dict = dict(enumerate(cw))

print("\nClass weights:")
for cls, w in zip(CLASS_NAMES, cw):
    print(f"  {cls:8s}: {w:.2f}")


# =============================================================================
# SECTION 7 — TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_SEED,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv,
    test_size=0.15 / 0.85,
    stratify=y_tv,
    random_state=RANDOM_SEED,
)

print(f"\nSplit → Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")


# =============================================================================
# SECTION 8 — MANUAL OVERSAMPLING OF MINORITY CLASSES
# =============================================================================
def oversample_minorities(X_df, y_arr, targets: dict, random_state=RANDOM_SEED):

    X_parts, y_parts = [X_df], [y_arr]
    for cls_idx, target_n in targets.items():
        mask   = (y_arr == cls_idx)
        n_have = mask.sum()
        if n_have >= target_n:
            continue  # already at or above target, skip
        X_min = X_df[mask]
        y_min = y_arr[mask]
        X_res, y_res = resample(
            X_min, y_min,
            replace=True,
            n_samples=target_n - n_have,  # only generate the extra needed
            random_state=random_state,
        )
        X_parts.append(X_res)
        y_parts.append(y_res)

    X_out = pd.concat(X_parts, ignore_index=True)
    y_out = np.concatenate(y_parts)
    return X_out, y_out


# Class indices: Move=0, Return=1, Stay=2, Unsure=3
OVERSAMPLE_TARGETS = {
    0: 300,   # Move:   193 → 300
    1: 250,   # Return:  85 → 250
}

X_train_os, y_train_os = oversample_minorities(
    X_train, y_train, OVERSAMPLE_TARGETS
)

print(f"\nAfter oversampling — Training set: {len(y_train_os)} rows")
for cls, n in zip(CLASS_NAMES, np.bincount(y_train_os)):
    print(f"  {cls:8s}: {n:5d}  ({n/len(y_train_os)*100:.1f}%)")


# =============================================================================
# SECTION 9 — HYPERPARAMETER TUNING
# =============================================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def tune_model(name, base_pipe, param_dist, n_iter=30):
    print(f"\n{'─'*55}\nTuning: {name}  (n_iter={n_iter})")
    t0 = time.time()
    search = RandomizedSearchCV(
        base_pipe,
        param_dist,
        n_iter=n_iter,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
    )
    search.fit(X_train_os, y_train_os)
    print(f"  Best CV macro-F1 : {search.best_score_:.4f}")
    print(f"  Best params      : {search.best_params_}")
    print(f"  Time             : {time.time()-t0:.1f}s")
    return search.best_estimator_


# ── Model 1: Logistic Regression (baseline) ──────────────────────────────────
lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight=cw_dict,
        random_state=RANDOM_SEED,
    )),
])
lr_params = {
    "clf__C": loguniform(0.001, 10),  # regularisation strength
}
best_lr = tune_model("Logistic Regression", lr_pipe, lr_params, n_iter=20)


# ── Model 2: Random Forest ────────────────────────────────────────────────────
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf",  RandomForestClassifier(
        class_weight=cw_dict,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )),
])
rf_params = {
    "clf__n_estimators":     randint(100, 500),     # number of trees
    "clf__max_depth":        [3, 5, 8, 10, None],   # max tree depth (None = unlimited)
    "clf__min_samples_leaf": randint(5, 40),         # min samples per leaf node
    "clf__max_features":     ["sqrt", "log2", 0.3], # features considered per split
}
best_rf = tune_model("Random Forest", rf_pipe, rf_params, n_iter=30)


# ── Model 3: Gradient Boosting ────────────────────────────────────────────────
gb_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf",  GradientBoostingClassifier(random_state=RANDOM_SEED)),
])
gb_params = {
    "clf__n_estimators":     randint(100, 400),      # number of boosting rounds
    "clf__max_depth":        [2, 3, 4, 5],           # shallow trees reduce overfitting
    "clf__learning_rate":    loguniform(0.01, 0.3),  # shrinkage parameter
    "clf__subsample":        uniform(0.6, 0.4),      # fraction of samples per tree
    "clf__min_samples_leaf": randint(5, 30),
}
best_gb = tune_model("Gradient Boosting", gb_pipe, gb_params, n_iter=30)



# =============================================================================
# SECTION 10 — CROSS-VALIDATION EVALUATION
# =============================================================================
print("\n" + "="*55)
print("CROSS-VALIDATION (tuned models, oversampled train)")
print("="*55)

scoring = ["f1_macro", "balanced_accuracy", "accuracy"]

tuned_models = {
    "Logistic Regression": best_lr,
    "Random Forest":       best_rf,
    "Gradient Boosting":   best_gb,
}

cv_summary = {}
for name, pipe in tuned_models.items():
    cv_res = cross_validate(
        pipe, X_train_os, y_train_os,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )
    cv_summary[name] = cv_res
    print(f"\n{name}")
    print(f"  Val  macro-F1 : {cv_res['test_f1_macro'].mean():.4f} ± {cv_res['test_f1_macro'].std():.4f}")
    print(f"  Train macro-F1: {cv_res['train_f1_macro'].mean():.4f} ± {cv_res['train_f1_macro'].std():.4f}")
    gap = cv_res['train_f1_macro'].mean() - cv_res['test_f1_macro'].mean()
    print(f"  Overfitting gap: {gap:.4f}  {'large gap' if gap > 0.3 else 'acceptable'}")


# =============================================================================
# SECTION 11 — FINAL FIT AND HOLD-OUT EVALUATION
# =============================================================================
print("\n" + "="*55)
print("HOLD-OUT RESULTS (validation & test sets)")
print("="*55)

fitted = {}
for name, pipe in tuned_models.items():
    pipe.fit(X_train_os, y_train_os)
    fitted[name] = pipe
    yv = pipe.predict(X_val)
    yt = pipe.predict(X_test)
    print(f"\n{'─'*50}")
    print(f"{name}")
    print(f"  Val  macro-F1 : {f1_score(y_val, yv, average='macro'):.4f}")
    print(f"  Test macro-F1 : {f1_score(y_test, yt, average='macro'):.4f}")
    print(classification_report(y_test, yt, target_names=CLASS_NAMES, digits=3))


# =============================================================================
# SECTION 12 — SUMMARY TABLE AND BEST MODEL SELECTION
# =============================================================================
print("\n" + "="*55)
print("SUMMARY TABLE — TEST SET")
print("="*55)

rows = []
for name, pipe in fitted.items():
    yt  = pipe.predict(X_test)
    mf1 = f1_score(y_test, yt, average="macro")
    ba  = balanced_accuracy_score(y_test, yt)
    acc = (y_test == yt).mean()
    rows.append({"Model": name, "Macro-F1": mf1, "Bal.Acc": ba, "Accuracy": acc})
    print(f"{name:<28}  Macro-F1={mf1:.4f}  BalAcc={ba:.4f}  Acc={acc:.4f}")

summary_df = pd.DataFrame(rows)

# Select best model by test macro-F1
best_name = summary_df.loc[summary_df["Macro-F1"].idxmax(), "Model"]
best_pipe  = fitted[best_name]
print(f"\n✓ Best model: {best_name}  "
      f"(Test macro-F1: {summary_df['Macro-F1'].max():.4f})")


# =============================================================================
# SECTION 13 — COUNTRY SUBGROUP ANALYSIS (Research Question 1.3)
# =============================================================================
print("\n" + "="*55)
print("COUNTRY SUBGROUP ANALYSIS (best model on test set)")
print("="*55)

countries = df.loc[X_test.index, "current_country"].values
country_results = {}

for ctry in sorted(set(countries)):
    mask = (countries == ctry)
    if mask.sum() < 10:
        continue   # skip countries with too few test observations
    y_c    = y_test[mask]
    X_c    = X_test[mask]
    y_pred = best_pipe.predict(X_c)
    mf1    = f1_score(y_c, y_pred, average="macro", zero_division=0)
    ba     = balanced_accuracy_score(y_c, y_pred)
    n      = mask.sum()
    country_results[ctry] = {"n": n, "Macro-F1": mf1, "Bal.Acc": ba}
    print(f"  {ctry:<20}  n={n:4d}  Macro-F1={mf1:.4f}  Bal.Acc={ba:.4f}")

# Per-class F1 by country (shows which intention categories drive country gaps)
print("\nPer-class F1 by country (Move / Return / Stay / Unsure):")
print(f"{'Country':<20} {'Move':>7} {'Return':>7} {'Stay':>7} {'Unsure':>7}")
for ctry in sorted(country_results):
    mask   = (countries == ctry)
    y_c    = y_test[mask]
    X_c    = X_test[mask]
    y_pred = best_pipe.predict(X_c)
    pc     = f1_score(y_c, y_pred, average=None,
                      zero_division=0, labels=[0, 1, 2, 3])
    print(f"{ctry:<20} {pc[0]:>7.3f} {pc[1]:>7.3f} {pc[2]:>7.3f} {pc[3]:>7.3f}")


# =============================================================================
# SECTION 14 — PERMUTATION IMPORTANCE
# =============================================================================
print(f"\nComputing permutation importance for {best_name}...")

prep_val = best_pipe.named_steps["prep"].transform(X_val)
clf_     = best_pipe.named_steps["clf"]

perm_res = permutation_importance(
    clf_, prep_val, y_val,
    n_repeats=15,
    random_state=RANDOM_SEED,
    scoring="f1_macro",
    n_jobs=-1,
)

# Collapse importance scores back from OHE dummies to original variable level
# (e.g. all gender_Male, gender_Female, gender_Unknown → "gender")
ohe       = best_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
ohe_names = ohe.get_feature_names_out(CAT_FEATURES).tolist()
feat_names = ohe_names + NUM_FEATURES
imp_means  = perm_res.importances_mean

var_imp = {}
for orig in CAT_FEATURES:
    mask_ = np.array([n.startswith(orig + "_") for n in feat_names])
    var_imp[orig] = imp_means[mask_].sum()
for n in NUM_FEATURES:
    var_imp[n] = imp_means[feat_names.index(n)]

# Human-readable labels for the plot
readable = {
    "temp_visit":           "Temp. visit intention",
    "legal_status":         "Legal status",
    "main_activity":        "Main activity",
    "valid_residency":      "Valid residency",
    "governorate":          "Origin governorate",
    "current_country":      "Host country",
    "years_since_arrival":  "Years since arrival",
    "needs_care":           "Care needs",
    "poses_document":       "Document possession",
    "age":                  "Age group",
    "family_size_num":      "Family size",
    "gender":               "Gender",
    "living_family_spouse": "Living w/ spouse",
    "living_family_child":  "Living w/ child",
    "living_family_old":    "Living w/ elderly",
}

vi_df = (
    pd.DataFrame.from_dict(var_imp, orient="index", columns=["importance"])
    .sort_values("importance")
)
vi_df.index = [readable.get(i, i) for i in vi_df.index]


# =============================================================================
# SECTION 15 — FIGURES
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})
COLORS       = ["#4C72B0", "#DD8452", "#55A868"]
LABELS_SHORT = ["Logistic\nRegression", "Random\nForest",
                "Gradient\nBoosting"]

# ── Fig 1: CV macro-F1 train vs validation ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, (name, cv_res) in enumerate(cv_summary.items()):
    v = cv_res["test_f1_macro"]
    t = cv_res["train_f1_macro"]
    ax.bar(i - 0.18, v.mean(), 0.32, color=COLORS[i], alpha=0.85,
           label=f"{LABELS_SHORT[i].replace(chr(10), ' ')} (val)",
           yerr=v.std(), capsize=4, error_kw={"elinewidth": 1.5})
    ax.bar(i + 0.18, t.mean(), 0.32, color=COLORS[i], alpha=0.35,
           label=f"{LABELS_SHORT[i].replace(chr(10), ' ')} (train)",
           yerr=t.std(), capsize=4, error_kw={"elinewidth": 1.5})
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(LABELS_SHORT)
ax.set_ylabel("Macro-averaged F1")
ax.set_ylim(0, 1.0)
ax.axhline(0.25, ls="--", color="gray", lw=1, label="Random baseline")
ax.set_title("5-fold CV: Train vs Validation Macro-F1\n(oversampled train, tuned models)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig1_cv_f1.png", dpi=150)
plt.close()

# ── Fig 2: Per-class F1 on test set ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(CLASS_NAMES))
w = 0.20
for i, (name, pipe) in enumerate(fitted.items()):
    pc = f1_score(y_test, pipe.predict(X_test), average=None)
    ax.bar(x + i * w, pc, w,
           label=LABELS_SHORT[i].replace("\n", " "),
           color=COLORS[i], alpha=0.85)
ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.0)
ax.set_title("Per-class F1 — Test Set")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig2_perclass_f1.png", dpi=150)
plt.close()

# ── Fig 3: Confusion matrix — best model ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}\n(Test Set)", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig3_confusion_matrix.png", dpi=150)
plt.close()

# ── Fig 4: Row-normalised confusion matrix (error rates) ─────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Row-normalised Confusion Matrix\n(proportion of each true class predicted as each class)")
for i in range(4):
    for j in range(4):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_norm[i,j]:.2f}\n(n={cm[i,j]})",
                ha="center", va="center", fontsize=9, color=color)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig4_normalised_confusion.png", dpi=150)
plt.close()

# ── Fig 5: Permutation importance ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_bar = ["#C44E52" if v > 0 else "#4C72B0" for v in vi_df["importance"]]
ax.barh(vi_df.index, vi_df["importance"], color=colors_bar, alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Permutation Importance (mean macro-F1 decrease on val set)")
ax.set_title(f"Permutation Importance — {best_name}")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig5_perm_importance.png", dpi=150)
plt.close()

# ── Fig 6: Country subgroup macro-F1 ─────────────────────────────────────────
ctry_names = list(country_results.keys())
ctry_f1    = [country_results[c]["Macro-F1"] for c in ctry_names]
ctry_n     = [country_results[c]["n"] for c in ctry_names]
order      = np.argsort(ctry_f1)
ctry_names = [ctry_names[i] for i in order]
ctry_f1    = [ctry_f1[i]    for i in order]
ctry_n     = [ctry_n[i]     for i in order]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(ctry_names, ctry_f1, color="#0D7377", alpha=0.85)
for bar, n, f in zip(bars, ctry_n, ctry_f1):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"F1={f:.3f}  (n={n})", va="center", fontsize=10)
ax.axvline(np.mean(ctry_f1), ls="--", color="gray", lw=1.2,
           label=f"Mean = {np.mean(ctry_f1):.3f}")
ax.set_xlabel("Macro-F1")
ax.set_xlim(0, max(ctry_f1) * 1.38)
ax.set_title(f"Country Subgroup Macro-F1 — {best_name} (Test Set)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig6_country_f1.png", dpi=150)
plt.close()

# ── Fig 7: CV fold stability ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, (name, cv_res) in enumerate(cv_summary.items()):
    ax.plot([1, 2, 3, 4, 5], cv_res["test_f1_macro"],
            marker="o",
            label=LABELS_SHORT[i].replace("\n", " "),
            color=COLORS[i],
            lw=2)
ax.axhline(0.25, ls="--", color="gray", lw=1, label="Random baseline")
ax.set_xlabel("Fold")
ax.set_ylabel("Macro-F1")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(0, 1.0)
ax.set_title("Macro-F1 per CV Fold — Stability Check")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig7_fold_stability.png", dpi=150)
plt.close()

print(f"✓ Best model: {best_name}  "
      f"(Test macro-F1: {summary_df['Macro-F1'].max():.4f})")