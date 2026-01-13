import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from sklearn.preprocessing import label_binarize

# ----------------------------
# 1) File paths for 5 classes
# ----------------------------
paths = {
    "static":        "imu_smooth_m0.csv",
    "shortdmbl":     "imu_smooth_m1.csv",
    "longdmbl":      "imu_smooth_m2.csv",
    "underdmbl":     "imu_smooth_m3.csv",
    "rotationdmbl":  "imu_smooth_m4.csv",
}

# CSVs have no header row; enforce column names:
colnames = ["timestamp","ax","ay","az","gx","gy","gz","roll","pitch","yaw"]
channels = ["ax","ay","az","gx","gy","gz","roll","pitch","yaw"]

dfs = {label: pd.read_csv(p, header=None, names=colnames) for label, p in paths.items()}

# ----------------------------
# 2) Windowing (100 Hz IMU)
# ----------------------------
FS = 100
WIN = 100      # 1 second window
STRIDE = 50    # 0.5 second overlap

def window_featurize(df, label, win=WIN, stride=STRIDE):
    """
    Convert a time-series file into windowed feature vectors.
    Features per channel: mean, std, min, max, mean-square (energy proxy)
    """
    X, y = [], []
    for start in range(0, len(df) - win + 1, stride):
        w = df.iloc[start:start+win]
        feats = []
        for c in channels:
            x = w[c].astype(float).to_numpy()
            feats.extend([
                x.mean(),
                x.std(ddof=0),
                x.min(),
                x.max(),
                np.mean(x**2)
            ])
        X.append(feats)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

X_list, y_list = [], []
for label, df in dfs.items():
    Xw, yw = window_featurize(df, label)
    X_list.append(Xw)
    y_list.append(yw)

X = np.vstack(X_list)
y = np.concatenate(y_list)

feature_names = [f"{c}_{stat}" for c in channels for stat in ["mean","std","min","max","msq"]]

print("Total windows:", X.shape[0], "Feature dimension:", X.shape[1])
print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

# ----------------------------
# 3) Split 60/20/20
# ----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Split sizes:", len(y_train), len(y_val), len(y_test))

# ----------------------------
# 4) Hyperparameter sweep (tune by validation log loss)
# ----------------------------
n_estimators_grid = [50, 100, 200]
max_depth_grid = [None, 3, 5, 8, 12]

best = None
records = []

for n_est in n_estimators_grid:
    for md in max_depth_grid:
        rf_try = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=md,
            random_state=42,
            n_jobs=-1
        )
        rf_try.fit(X_train, y_train)
        val_proba = rf_try.predict_proba(X_val)
        ll = log_loss(y_val, val_proba, labels=rf_try.classes_)

        records.append({"n_estimators": n_est, "max_depth": str(md), "val_log_loss": ll})
        if best is None or ll < best["val_log_loss"]:
            best = {"n_estimators": n_est, "max_depth": md, "val_log_loss": ll}

tune_df = pd.DataFrame(records).sort_values("val_log_loss").reset_index(drop=True)
print("\nTop sweep results:\n", tune_df.head(10))
print("\nBest params:", best)

# Train final RF
rf = RandomForestClassifier(
    n_estimators=best["n_estimators"],
    max_depth=best["max_depth"],
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ----------------------------
# 5) Test metrics
# ----------------------------
y_pred = rf.predict(X_test)
proba = rf.predict_proba(X_test)
labels = list(rf.classes_)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
ll   = log_loss(y_test, proba, labels=labels)

print("\n=== Test Metrics (Macro Avg) ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("Log Loss :", ll)

cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nConfusion matrix (rows=true, cols=pred), labels =", labels)
print(cm)

# ----------------------------
# 6) Save visuals (must include '_rf_')
# ----------------------------
out_dir = ""
roc_path  = os.path.join(out_dir, "roc_auc_rf_test.png")
ll_path   = os.path.join(out_dir, "logloss_rf_sweep.png")
imp_path  = os.path.join(out_dir, "feature_importance_rf.png")
tree_path = os.path.join(out_dir, "one_tree_rf.png")

# ROC-AUC curves (multiclass one-vs-rest)
y_test_bin = label_binarize(y_test, classes=labels)

plt.figure()
for i, lab in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest) - Random Forest - Test Set")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(roc_path, dpi=200)
plt.show()

# Log loss curve for sweep results
plt.figure()
for n_est in n_estimators_grid:
    sub = tune_df[tune_df["n_estimators"] == n_est].copy()

    def md_to_num(v):
        return 999 if v == "None" else int(v)

    sub["md_num"] = sub["max_depth"].apply(md_to_num)
    sub = sub.sort_values("md_num")

    plt.plot(sub["md_num"], sub["val_log_loss"], marker="o", label=f"n_estimators={n_est}")

plt.xlabel("max_depth (None shown as 999)")
plt.ylabel("Validation Log Loss")
plt.title("RF Validation Log Loss vs max_depth (sweep)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ll_path, dpi=200)
plt.show()

# Feature importances (Top 15)
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 5))
plt.bar(range(len(idx)), importances[idx])
plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
plt.title("Random Forest Feature Importances (Top 15)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(imp_path, dpi=200)
plt.show()

# Visualize one representative tree (tree #0) from the forest
est0 = rf.estimators_[0]
plt.figure(figsize=(22, 10))
plot_tree(
    est0,
    feature_names=feature_names,
    class_names=labels,
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=4
)
plt.title("One Tree from the Random Forest (Top Levels)")
plt.tight_layout()
plt.savefig(tree_path, dpi=200)
plt.show()

print("\nSaved figures:")
print(" ", roc_path)
print(" ", ll_path)
print(" ", imp_path)
print(" ", tree_path)
