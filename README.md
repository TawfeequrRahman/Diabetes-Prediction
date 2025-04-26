🩺 Disease Prediction Using Custom Gaussian Naive Bayes Classifier

This project implements a Disease Prediction System using a Gaussian Naive Bayes Classifier developed completely from scratch — without using ready-made ML models from libraries like scikit-learn.
It includes full preprocessing, manual model building, evaluation, and visualization steps.

📂 Table of Contents
About the Project
Project Pipeline
Technologies Used
Getting Started
Results

📜 About the Project
In this project, we predict the likelihood of a patient having a disease based on their health parameters (such as Glucose level, BMI, Blood Pressure, etc.).
The Gaussian Naive Bayes algorithm is manually implemented to understand every step involved in building a machine learning model.

🔥 Project Pipeline

✅ Load Dataset
✅ Data Preprocessing:

Replace missing (zero) values with feature-wise means
Manual feature standardization

✅ Train-Test Split:

Custom random split without scikit-learn functions

✅ Model Development:

Calculation of mean, variance, and priors per class
Probability density function (Gaussian) implementation
Prediction by maximizing posterior probabilities

✅ Evaluation:

Accuracy, Precision, Recall, F1-score
Confusion Matrix
ROC Curve with AUC
Precision-Recall Curve

✅ Visualization:

Correlation heatmap
Histograms of feature distributions
Confusion matrix, ROC, and Precision-Recall curves
🛠️ Technologies Used

Python (Core language)
Pandas (Data manipulation)
NumPy (Numerical computations)
Matplotlib & Seaborn (Visualization)
scikit-learn (Only for ROC and PR curve plotting)

🚀 Getting Started

Prerequisites
Make sure you have Python installed. Then install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn
Running the Project

Clone the repository:
git clone https://github.com/your-username/disease-prediction-gaussian-nb.git
cd disease-prediction-gaussian-nb

Run the Python script:
python app.py

📈 Results

Achieved competitive Accuracy, Precision, Recall, and F1-score.
Clear visual insights through correlation heatmaps and metric curves.
Built a fully explainable model without any high-level ML libraries!
