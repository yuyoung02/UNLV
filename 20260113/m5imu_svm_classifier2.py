import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)

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
C_grid = [0.1, 1, 10, 100]
gamma_grid = ["scale", 0.01, 0.1, 1]

records = []
best = None

for C in C_grid:
    for gamma in gamma_grid:
        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42))
        ])
        svm.fit(X_train, y_train)
        val_proba = svm.predict_proba(X_val)
        ll = log_loss(y_val, val_proba, labels=svm.named_steps["clf"].classes_)
        records.append({"C": C, "gamma": str(gamma), "val_log_loss": ll})
        if best is None or ll < best["val_log_loss"]:
            best = {"C": C, "gamma": gamma, "val_log_loss": ll}

records = pd.DataFrame(records).sort_values("val_log_loss").reset_index(drop=True)
print("\nTop sweep results:\n", records.head(10))
print("\nBest params:", best)

# Train best SVM
svm_best = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], probability=True, random_state=42))
])
svm_best.fit(X_train, y_train)

# ----------------------------
# 5) Test metrics
# ----------------------------
y_pred = svm_best.predict(X_test)
proba = svm_best.predict_proba(X_test)
class_order = svm_best.named_steps["clf"].classes_.tolist()

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
ll   = log_loss(y_test, proba, labels=class_order)

print("\n=== Test Metrics (Macro Avg) ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("Log Loss :", ll)

cm = confusion_matrix(y_test, y_pred, labels=class_order)
print("\nConfusion matrix (rows=true, cols=pred), labels =", class_order)
print(cm)

# ----------------------------
# 6) Save visuals (must include '_svm_')
# ----------------------------
out_dir = ""
roc_path  = os.path.join(out_dir, "roc_auc_svm_test.png")
ll_path   = os.path.join(out_dir, "logloss_svm_heatmap.png")
viz_path  = os.path.join(out_dir, "decision_regions_svm_pca_svm.png")  # includes _svm_

# ROC-AUC curves (multiclass one-vs-rest)
y_test_bin = label_binarize(y_test, classes=class_order)

plt.figure()
for i, lab in enumerate(class_order):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest) - SVM - Test Set")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(roc_path, dpi=200)
plt.show()

# Log loss heatmap over (C, gamma)
gamma_labels = [str(g) for g in gamma_grid]
C_labels = [str(c) for c in C_grid]
heat = np.zeros((len(C_grid), len(gamma_grid)), dtype=float)

def gamma_to_idx(g):
    if g == "scale":
        return gamma_grid.index("scale")
    return gamma_grid.index(float(g))

for _, r in records.iterrows():
    i = C_grid.index(float(r["C"]))
    j = gamma_to_idx(str(r["gamma"]))
    heat[i, j] = float(r["val_log_loss"])

plt.figure(figsize=(7, 4))
plt.imshow(heat, interpolation="nearest")
plt.xticks(range(len(gamma_grid)), gamma_labels)
plt.yticks(range(len(C_grid)), C_labels)
plt.xlabel("gamma")
plt.ylabel("C")
plt.title("Validation Log Loss Heatmap (SVM RBF)")
for i in range(len(C_grid)):
    for j in range(len(gamma_grid)):
        plt.text(j, i, f"{heat[i,j]:.3f}", ha="center", va="center")
plt.colorbar(label="Log Loss")
plt.tight_layout()
plt.savefig(ll_path, dpi=200)
plt.show()

# Visualize the SVM model: decision regions in 2D PCA space (for teaching)
le = LabelEncoder()
y_train_num = le.fit_transform(y_train)

scaler = svm_best.named_steps["scaler"]
X_train_s = scaler.transform(X_train)

pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_train_s)

viz_clf = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"])
viz_clf.fit(X2, y_train_num)

x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))
Z = viz_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z.astype(float), alpha=0.25)
for k, lab in enumerate(le.classes_):
    mask = (y_train_num == k)
    plt.scatter(X2[mask, 0], X2[mask, 1], s=10, label=lab)
plt.title("SVM Decision Regions in 2D PCA Space (Visualization)")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(viz_path, dpi=200)
plt.show()

print("\nSaved figures:")
print(" ", roc_path)
print(" ", ll_path)
print(" ", viz_path)
