import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_curve, auc

# 1. Load Data
# Define file paths and class names
files = {
    'static': 'imu_smooth_m0.csv',
    'shortdmbl': 'imu_smooth_m1.csv',
    'longdmbl': 'imu_smooth_m2.csv',
    'underdmbl': 'imu_smooth_m3.csv',
    'rotationdmbl': 'imu_smooth_m4.csv'
}
cols = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']

# Load and label datasets
data_list = []
class_names = ['static', 'shortdmbl', 'longdmbl', 'underdmbl', 'rotationdmbl']

for i, class_name in enumerate(class_names):
    df = pd.read_csv(files[class_name], header=None, names=cols)
    df['label'] = i  # Assign numeric label 0-4
    data_list.append(df)

# Concatenate into a single dataframe
full_data = pd.concat(data_list, ignore_index=True)

# 2. Preprocessing
# Separate features (X) and target (y)
# Exclude 'timestamp' (col 0) and 'label' (last col) from features
X = full_data.iloc[:, 1:-1].values
y = full_data['label'].values

# Split Data: 60% Train, 20% Validation, 20% Test
# First, split into 80% (Train+Val) and 20% (Test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Second, split the 80% (Train+Val) into 75% Train (which is 60% of total) and 25% Val (which is 20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
# Initialize MLP Classifier
# max_iter=1 and warm_start=True allow us to loop manually to record log loss history
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                    alpha=0.0001, batch_size=64, learning_rate_init=0.001,
                    random_state=42, warm_start=True, max_iter=1)

train_loss_history = []
val_loss_history = []
epochs = 50
unique_classes = np.unique(y)

# Manual training loop to capture loss per epoch
for epoch in range(epochs):
    mlp.partial_fit(X_train_scaled, y_train, classes=unique_classes)

    # Calculate Log Loss for learning curves
    y_train_prob = mlp.predict_proba(X_train_scaled)
    y_val_prob = mlp.predict_proba(X_val_scaled)

    train_loss_history.append(log_loss(y_train, y_train_prob))
    val_loss_history.append(log_loss(y_val, y_val_prob))

# 4. Evaluation Metrics
y_test_pred = mlp.predict(X_test_scaled)
y_test_prob = mlp.predict_proba(X_test_scaled)

print("Evaluation Metrics (Test Set):")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

# 5. Visualizations

# --- Plot 1: Log Loss Curve ---
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Training Log Loss')
plt.plot(val_loss_history, label='Validation Log Loss')
plt.title('Log Loss Curve over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.savefig('log_loss_curve_mlp.png')
plt.show()

# --- Plot 2: ROC-AUC Curve ---
# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=unique_classes)
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

# Calculate ROC per class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'purple', 'orange']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC-AUC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_auc_curve_mlp.png')
plt.show()


# --- Plot 3: MLP Architecture Visualization ---
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Helper function to draw a neural network graph
    '''
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k', alpha=0.1)
                ax.add_artist(line)


fig = plt.figure(figsize=(10, 8))
ax = fig.gca()
ax.axis('off')
# Visualize architecture: Input(9) -> Hidden(64) -> Hidden(32) -> Output(5)
draw_neural_net(ax, .1, .9, .1, .9, [9, 64, 32, 5])
plt.title('MLP Neural Network Architecture Visualization')
plt.savefig('architecture_visualization_mlp.png')
plt.show()