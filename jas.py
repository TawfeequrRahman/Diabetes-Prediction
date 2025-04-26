import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from collections import Counter
import streamlit as st

st.set_page_config(page_title="KNN Diabetes Predictor", layout="wide")

# Upload widget OUTSIDE the cached function
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Step 1: Load the Diabetes Dataset
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                   'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
    df['BMI_Age'] = df['BMI'] * df['Age']
    return df

df = load_data(uploaded)
st.title("🧬 KNN Classifier from Scratch - Diabetes Detection")
st.markdown("This app implements KNN from scratch with interactive controls and visualizations.")

# Dataset filtering
with st.expander("🔍 Filter Dataset"):
    filter_col = st.selectbox("Filter by column", df.columns)
    if pd.api.types.is_numeric_dtype(df[filter_col]):
        min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
        val_range = st.slider("Range", min_val, max_val, (min_val, max_val))
        df = df[(df[filter_col] >= val_range[0]) & (df[filter_col] <= val_range[1])]
    else:
        selected = st.multiselect("Select values", df[filter_col].unique())
        if selected:
            df = df[df[filter_col].isin(selected)]

    st.dataframe(df.head(20))

# Preprocessing
X_raw_df = df.drop(columns=['Outcome']).copy()
X_raw_df['BMI_Age'] = X_raw_df['BMI'] * X_raw_df['Age']
X_raw_df = X_raw_df[X_raw_df.columns]
X_raw = X_raw_df.values
y = df['Outcome'].values

# Standardization functions with saved mean/std
mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0)

def standardize(X):
    return (X - mean) / std

X = standardize(X_raw)

# Oversample Class 1
class_0 = X[y == 0]
class_1 = X[y == 1]
repeat_factor = len(class_0) // len(class_1)
X_oversampled = np.concatenate([class_0, np.tile(class_1, (repeat_factor, 1))], axis=0)
y_oversampled = np.concatenate([np.zeros(len(class_0)), np.ones(len(class_1) * repeat_factor)]).astype(int)
shuffle = np.random.permutation(len(X_oversampled))
X_resampled, y_resampled = X_oversampled[shuffle], y_oversampled[shuffle]

# Train-test split
def train_test_split(X, y, test_size=0.2):
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

# KNN Functions
def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = [euclidean(x_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    return Counter(k_labels).most_common(1)[0][0], distances, k_indices

def knn_model(X_train, y_train, X_test, k):
    return np.array([knn_predict(X_train, y_train, x, k)[0] for x in X_test])

def cross_val_score(X, y, k, folds=5):
    size = len(X) // folds
    scores = []
    for i in range(folds):
        start, end = i * size, (i + 1) * size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        pred = knn_model(X_train, y_train, X_val, k)
        scores.append(np.mean(pred == y_val))
    return np.mean(scores)

# Sidebar
st.sidebar.header("⚙️ Model Settings")
k = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)

# User Input Section
st.sidebar.header("📥 Manual Prediction")
user_input = {}
columns = X_raw_df.columns
for col in columns:
    user_input[col] = st.sidebar.number_input(f"{col}", min_value=0.0, value=float(df[col].mean()), key=f"user_{col}")
user_input_df = pd.DataFrame([user_input])
user_input_df = user_input_df[X_raw_df.columns]

# Standardize User Input
if user_input_df.shape[1] == mean.shape[0]:
    user_input_std = standardize(user_input_df.values)
    y_user_pred, user_distances, user_k_indices = knn_predict(X_train, y_train, user_input_std[0], k)
    neighbor_confidence = np.mean(y_train[user_k_indices])
    st.sidebar.write(f"### 🧾 Prediction: {'🩸 Diabetic' if y_user_pred == 1 else '✅ Non-Diabetic'}")
    st.sidebar.write("### 🔍 Confidence:", f"{neighbor_confidence:.2f}")

    fig_bar = px.bar(x=list(range(1, k + 1)),
                     y=y_train[user_k_indices],
                     labels={'x': 'Neighbor Rank', 'y': 'Label'},
                     title='Top K Neighbor Labels')
    st.sidebar.plotly_chart(fig_bar, use_container_width=True)

    fig_dist = px.line(x=list(range(1, k + 1)),
                       y=np.sort(user_distances)[:k],
                       labels={'x': 'Neighbor Rank', 'y': 'Distance'},
                       title='Top K Distances to User Input')
    st.sidebar.plotly_chart(fig_dist, use_container_width=True)
else:
    st.sidebar.error("❗ User input shape mismatch. Check input feature count.")

# Predict
y_pred = knn_model(X_train, y_train, X_test, k)

# Metrics
def metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return accuracy, precision, recall, f1, np.array([[tn, fp], [fn, tp]])

acc, pre, rec, f1, cm = metrics(y_test, y_pred)

with st.container():
    st.subheader("📊 Model Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{acc:.2f}", help="How often the classifier is correct")
    with col2:
        st.metric("Precision", f"{pre:.2f}", help="Proportion of predicted positives that are real")
    with col3:
        st.metric("Recall", f"{rec:.2f}", help="Proportion of actual positives identified correctly")
    with col4:
        st.metric("F1 Score", f"{f1:.2f}", help="Balance between precision and recall")

# Tabs
conf_tab, acc_tab, roc_tab, imp_tab, pr_tab, tsne_tab, lime_tab = st.tabs([
    "Confusion Matrix", "K vs Accuracy", "ROC Curve", "Feature Importance", "Precision-Recall", "t-SNE Projection", "🔎 LIME Explanation"])

with conf_tab:
    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=["Non-Diabetic", "Diabetic"],
                       y=["Non-Diabetic", "Diabetic"],
                       text_auto=True,
                       color_continuous_scale="blues")
    st.plotly_chart(fig_cm, use_container_width=True)

with acc_tab:
    st.subheader("K vs Accuracy")
    k_values = list(range(1, 16))
    acc_values = [cross_val_score(X_train, y_train, k_val) for k_val in k_values]
    fig_kacc = go.Figure()
    fig_kacc.add_trace(go.Scatter(x=k_values, y=acc_values, mode='lines+markers'))
    fig_kacc.update_layout(title="Accuracy vs K", xaxis_title="K", yaxis_title="Cross-Validation Accuracy")
    st.plotly_chart(fig_kacc, use_container_width=True)

with roc_tab:
    st.subheader("ROC Curve")
    fpr = []
    tpr = []
    thresholds = np.linspace(0, 1, 100)
    for thresh in thresholds:
        preds = (y_pred >= thresh).astype(int)
        _, _, r, _, cm_tmp = metrics(y_test, preds)
        tpr.append(r)
        fpr.append(cm_tmp[0][1] / (cm_tmp[0][0] + cm_tmp[0][1] + 1e-6))
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', fill='tozeroy'))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

with imp_tab:
    st.subheader("Feature Importance (via Accuracy Drop)")
    baseline_acc = np.mean(y_pred == y_test)
    importances = []
    for i in range(X.shape[1]):
        X_train_drop = np.delete(X_train, i, axis=1)
        X_test_drop = np.delete(X_test, i, axis=1)
        y_pred_drop = knn_model(X_train_drop, y_train, X_test_drop, k)
        drop_acc = np.mean(y_pred_drop == y_test)
        importances.append(baseline_acc - drop_acc)
    fig_imp = px.bar(x=X_raw_df.columns, y=importances, labels={'x': 'Feature', 'y': 'Accuracy Drop'},
                     title="Leave-One-Feature-Out Accuracy Drop")
    st.plotly_chart(fig_imp, use_container_width=True)

with pr_tab:
    st.subheader("Precision-Recall Curve")
    thresholds = np.linspace(0, 1, 100)
    precision_curve = []
    recall_curve = []
    for thresh in thresholds:
        preds = (y_pred >= thresh).astype(int)
        _, p, r, _, _ = metrics(y_test, preds)
        precision_curve.append(p)
        recall_curve.append(r)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines+markers'))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True)

with tsne_tab:
    try:
        from openTSNE import TSNE
        st.sidebar.subheader("🧭 t-SNE Parameters")
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
        n_iter = st.sidebar.slider("Iterations", 250, 1000, 300, step=50)
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        X_embedded = tsne.fit(X_resampled).view(np.ndarray)
        tsne_df = pd.DataFrame(X_embedded, columns=['Dim1', 'Dim2'])
        tsne_df['Outcome'] = y_resampled

        fig1 = px.scatter(tsne_df, x='Dim1', y='Dim2', color=tsne_df['Outcome'].map({0: "Non-Diabetic", 1: "Diabetic"}),
                          title="t-SNE Projection by Class", color_discrete_map={"Non-Diabetic": "blue", "Diabetic": "red"})

        fig2 = px.scatter(tsne_df, x='Dim1', y='Dim2', color=tsne_df['Outcome'],
                          title="t-SNE Projection (Raw Labels)", color_continuous_scale='Viridis')

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

    except ImportError:
        st.warning("Install openTSNE using `pip install openTSNE` to enable t-SNE projection without sklearn.")