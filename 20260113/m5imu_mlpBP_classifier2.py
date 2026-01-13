import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)

# ---------------------------------------------------------
# 1) Load your 5 class CSVs (NO headers assumed)
# ---------------------------------------------------------
paths = {
    "static":        "imu_smooth_m0.csv",
    "shortdmbl":     "imu_smooth_m1.csv",
    "longdmbl":      "imu_smooth_m2.csv",
    "underdmbl":     "imu_smooth_m3.csv",
    "rotationdmbl":  "imu_smooth_m4.csv",
}

colnames  = ["timestamp","ax","ay","az","gx","gy","gz","roll","pitch","yaw"]
channels  = ["ax","ay","az","gx","gy","gz","roll","pitch","yaw"]

dfs = {label: pd.read_csv(p, header=None, names=colnames) for label, p in paths.items()}

# ---------------------------------------------------------
# 2) Windowed feature extraction (recommended for 100 Hz IMU)
# ---------------------------------------------------------
FS = 100
WIN = 100     # 1 second window
STRIDE = 50   # 0.5 sec overlap

def window_featurize(df, label, win=WIN, stride=STRIDE):
    """
    For each window, compute per-channel features:
    mean, std, min, max, mean-square (energy proxy)
    """
    X, y = [], []
    for start in range(0, len(df) - win + 1, stride):
        w = df.iloc[start:start+win]
        feats = []
        for c in channels:
            x = w[c].astype(float).to_numpy()
            feats.extend([x.mean(), x.std(ddof=0), x.min(), x.max(), np.mean(x**2)])
        X.append(feats)
        y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y)

X_list, y_list = [], []
for label, df in dfs.items():
    Xw, yw = window_featurize(df, label)
    X_list.append(Xw)
    y_list.append(yw)

X = np.vstack(X_list)
y = np.concatenate(y_list)

feature_names = [f"{c}_{stat}" for c in channels for stat in ["mean","std","min","max","msq"]]

print("Total windows:", X.shape[0], "Feature dim:", X.shape[1])
print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

# ---------------------------------------------------------
# 3) Split 60% train, 20% val, 20% test (stratified)
# ---------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Encode labels to integers for ROC & log-loss tracking
le = LabelEncoder()
y_train_i = le.fit_transform(y_train)
y_val_i   = le.transform(y_val)
y_test_i  = le.transform(y_test)
class_names = le.classes_.tolist()
n_classes = len(class_names)

# Scale features (important for MLP)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ---------------------------------------------------------
# 4) MLP + Backprop training (explicit epoch loop using partial_fit)
#    - This makes the "Log Loss vs Epochs" curve clear for students.
# ---------------------------------------------------------
# MLPClassifier uses backpropagation internally (gradient-based optimization).
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=1,         # one "epoch" per partial_fit call
    warm_start=True,
    random_state=42
)

EPOCHS = 120
PATIENCE = 12

train_ll, val_ll = [], []
best_val = float("inf")
best_epoch = 0
best_params = None
pat = 0

for epoch in range(1, EPOCHS + 1):
    # First call needs 'classes='
    if epoch == 1:
        mlp.partial_fit(X_train_s, y_train_i, classes=np.arange(n_classes))
    else:
        mlp.partial_fit(X_train_s, y_train_i)

    # Track log-loss on train/val
    p_tr = mlp.predict_proba(X_train_s)
    p_va = mlp.predict_proba(X_val_s)
    ll_tr = log_loss(y_train_i, p_tr, labels=np.arange(n_classes))
    ll_va = log_loss(y_val_i, p_va, labels=np.arange(n_classes))

    train_ll.append(ll_tr)
    val_ll.append(ll_va)

    # Early stopping
    if ll_va + 1e-6 < best_val:
        best_val = ll_va
        best_epoch = epoch
        best_params = mlp.coefs_, mlp.intercepts_
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
            break

# Restore best weights (so evaluation uses best validation point)
if best_params is not None:
    mlp.coefs_, mlp.intercepts_ = best_params

print(f"Stopped at epoch={len(train_ll)}; best epoch={best_epoch}, best val logloss={best_val:.4f}")

# ---------------------------------------------------------
# 5) Evaluate on test set: Precision/Recall/Accuracy/F1
# ---------------------------------------------------------
y_pred = mlp.predict(X_test_s)
p_test = mlp.predict_proba(X_test_s)

acc  = accuracy_score(y_test_i, y_pred)
prec = precision_score(y_test_i, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test_i, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test_i, y_pred, average="macro", zero_division=0)
cm   = confusion_matrix(y_test_i, y_pred)

print("\n=== Test Metrics (Macro Avg) ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)

print("\nConfusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=[f"true_{c}" for c in class_names],
                      columns=[f"pred_{c}" for c in class_names]))

# ---------------------------------------------------------
# 6) Save required visualizations with substring "_mlpBP_"
# ---------------------------------------------------------
out_dir = ""
roc_path  = os.path.join(out_dir, "roc_auc_mlpBP_test.png")
ll_path   = os.path.join(out_dir, "logloss_mlpBP_epochs.png")
arch_path = os.path.join(out_dir, "mlpBP_architecture_mlpBP.png")
w1_path   = os.path.join(out_dir, "first_layer_weights_mlpBP.png")

# (A) ROC-AUC (One-vs-Rest)
y_test_bin = label_binarize(y_test_i, classes=np.arange(n_classes))

plt.figure()
for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], p_test[:, i])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (OvR) - MLP-BP - Test Set")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(roc_path, dpi=200)
plt.show()

# (B) Log Loss vs Epochs
plt.figure()
plt.plot(range(1, len(train_ll)+1), train_ll, label="Train log loss")
plt.plot(range(1, len(val_ll)+1), val_ll, label="Val log loss")
plt.axvline(best_epoch, linestyle="--", label=f"Best epoch={best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("MLP-BP Log Loss vs Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ll_path, dpi=200)
plt.show()

# (C) Model visualization: Architecture diagram (high-level)
# Show up to 10 nodes per layer for readability
input_dim = X.shape[1]
hidden = (64, 32)
output_dim = n_classes

plt.figure(figsize=(9, 4))
layers = [("Input", min(10, input_dim)),
          ("H1", min(10, hidden[0])),
          ("H2", min(10, hidden[1])),
          ("Output", output_dim)]
xpos = np.linspace(0.1, 0.9, len(layers))

for x, (lname, n) in zip(xpos, layers):
    ys = np.linspace(0.1, 0.9, n)
    for yv in ys:
        plt.scatter([x], [yv], s=120)
    shown = input_dim if lname == "Input" else (hidden[0] if lname=="H1" else (hidden[1] if lname=="H2" else output_dim))
    plt.text(x, 0.95, f"{lname}\n({shown})", ha="center", va="bottom")

for li in range(len(layers)-1):
    x1, x2 = xpos[li], xpos[li+1]
    y1s = np.linspace(0.1, 0.9, layers[li][1])
    y2s = np.linspace(0.1, 0.9, layers[li+1][1])
    for a in y1s[:min(4,len(y1s))]:
        for b in y2s[:min(4,len(y2s))]:
            plt.plot([x1, x2], [a, b], linewidth=0.8, alpha=0.5)

plt.axis("off")
plt.title("MLP-BP Architecture (high-level visualization)")
plt.tight_layout()
plt.savefig(arch_path, dpi=200)
plt.show()

# (D) Model-derived visualization: First-layer weight heatmap
# coefs_[0] has shape (input_dim, hidden1); transpose to (hidden1, input_dim)
W1 = mlp.coefs_[0].T
plt.figure(figsize=(10, 4))
plt.imshow(W1, aspect="auto")
plt.colorbar(label="Weight value")
plt.xlabel("Input feature index")
plt.ylabel("Hidden neuron index (H1)")
plt.title("First Layer Weights Heatmap (MLP-BP)")
plt.tight_layout()
plt.savefig(w1_path, dpi=200)
plt.show()

print("\nSaved figures:")
print(" ", roc_path)
print(" ", ll_path)
print(" ", arch_path)
print(" ", w1_path)
