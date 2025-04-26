# === Import Basic Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt, pi, exp
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# === Load Dataset ===
df = pd.read_csv('/Users/tawfeequrrahman/vs_code/NN_ML/disease_data.csv')

# === Preprocessing ===
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Replace 0s with NaNs in specific columns and fill with mean
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X[cols_with_zeros] = X[cols_with_zeros].replace(0, np.nan)
X.fillna(X.mean(), inplace=True)

# === Correlation Heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# === Feature Distributions ===
X.hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# === Standardization (manually) ===
X = (X - X.mean()) / X.std()

# === Train-Test Split (manual) ===
def train_test_split_manual(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y)

# === Naive Bayes Classifier From Scratch ===
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.mean = {}
        self.var = {}
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.mean[c] = X_c.mean().values
            self.var[c] = X_c.var().values

    def gaussian_prob(self, x, mean, var):
        eps = 1e-6
        coeff = 1.0 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(- (x - mean)**2 / (2 * var + eps))
        return coeff * exponent

    def predict_instance(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_likelihood = np.sum(np.log(self.gaussian_prob(x, self.mean[c], self.var[c])))
            posteriors[c] = prior + class_likelihood
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self.predict_instance(x) for x in X.values])
    
    def predict_proba(self, X):
        probas = []
        for x in X.values:
            class_probs = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian_prob(x, self.mean[c], self.var[c])))
                class_probs[c] = prior + likelihood
            # Softmax for probabilities
            max_log = max(class_probs.values())
            exp_probs = {k: np.exp(v - max_log) for k, v in class_probs.items()}
            total = sum(exp_probs.values())
            probas.append([exp_probs[0] / total, exp_probs[1] / total])
        return np.array(probas)

# === Train and Predict ===
model = GaussianNaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# === Evaluation Metrics ===
def evaluate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return accuracy, precision, recall, f1, tp, tn, fp, fn

accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate(y_test.to_numpy(), y_pred)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# === Confusion Matrix ===
cm = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# === Precision-Recall Curve ===
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
