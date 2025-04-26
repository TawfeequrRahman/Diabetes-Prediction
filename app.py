import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sys

# Check if script is run directly instead of via streamlit CLI
if __name__ == "__main__":
    if not hasattr(st, 'runtime'):
        print("Warning: It is recommended to run this app using the command:")
        print("    streamlit run app.py")
        print("Running directly with python may cause warnings and missing Streamlit features.")
        # Optionally, exit here to prevent running without Streamlit context
        # sys.exit(1)

# === Naive Bayes Classifier ===
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
            likelihood = np.sum(np.log(self.gaussian_prob(x, self.mean[c], self.var[c])))
            posteriors[c] = prior + likelihood
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
            max_log = max(class_probs.values())
            exp_probs = {k: np.exp(v - max_log) for k, v in class_probs.items()}
            total = sum(exp_probs.values())
            probas.append([exp_probs[0] / total, exp_probs[1] / total])
        return np.array(probas)

# === Load & Preprocess Data ===
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/tawfeequrrahman/vs_code/NN_ML/disease_data.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[cols_with_zeros] = X[cols_with_zeros].replace(0, np.nan)
    X.fillna(X.mean(), inplace=True)
    return X, y

def preprocess(X):
    return (X - X.mean()) / X.std()

def train_test_split_manual(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def evaluate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return accuracy, precision, recall, f1, np.array([[tn, fp], [fn, tp]])

# === Streamlit UI ===
st.title("🧠 Disease Prediction System (Naive Bayes)")

X, y = load_data()
X_std = preprocess(X)
X_train, X_test, y_train, y_test = train_test_split_manual(X_std, y)

model = GaussianNaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# === Evaluation ===
acc, prec, rec, f1, cm = evaluate(y_test.to_numpy(), y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {acc:.4f}  \n**Precision:** {prec:.4f}  \n**Recall:** {rec:.4f}  \n**F1 Score:** {f1:.4f}")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], linestyle='--')
ax_roc.set_title("ROC Curve")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# === PR Curve ===
prec_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
fig_pr, ax_pr = plt.subplots()
ax_pr.plot(recall_vals, prec_vals)
ax_pr.set_title("Precision-Recall Curve")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
st.pyplot(fig_pr)

# === Correlation Heatmap & Feature Dist ===
with st.expander("🔍 Feature Insights"):
    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("Feature Distributions")
    fig_dist, ax_dist = plt.subplots(figsize=(12, 10))
    # Plot histograms for each column individually to avoid clearing the figure
    for i, col in enumerate(X.columns):
        ax = fig_dist.add_subplot(4, 3, i + 1)  # Adjust grid size as needed
        ax.hist(X[col], bins=20)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig_dist)

# === User Prediction Input ===
st.subheader("🔮 Predict from Input")
input_vals = {}
for col in X.columns:
    input_vals[col] = st.number_input(f"{col}", value=float(X[col].mean()), step=0.1)

if st.button("Predict"):
    user_df = pd.DataFrame([input_vals])
    user_std = (user_df - X.mean()) / X.std()
    result = model.predict(user_std)[0]
    prob = model.predict_proba(user_std)[0][1]
    st.write(f"### Prediction: {'Positive' if result == 1 else 'Negative'}")
    st.write(f"Probability of Disease: {prob:.2%}")

# === Batch Prediction via CSV ===
st.subheader("📁 Batch Prediction (Upload CSV)")
csv_file = st.file_uploader("Upload a CSV file with same columns as dataset", type=['csv'])

if csv_file:
    batch_data = pd.read_csv(csv_file)
    batch_data.fillna(X.mean(), inplace=True)
    batch_std = (batch_data - X.mean()) / X.std()
    batch_preds = model.predict(batch_std)
    batch_probs = model.predict_proba(batch_std)[:, 1]
    batch_data['Prediction'] = batch_preds
    batch_data['Probability'] = batch_probs
    st.dataframe(batch_data)
    st.download_button("Download Predictions", batch_data.to_csv(index=False), "predictions.csv", "text/csv")
