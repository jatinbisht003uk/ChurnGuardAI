ChurnGuard AI — Customer Churn Prediction & Retention Analytics

ChurnGuard AI is a machine learning application that predicts customer churn risk and provides analytics to support customer retention decisions.

Features

* Customer Churn Prediction: Estimate an individual customer’s churn probability.
* Business Analytics Dashboard: Explore customer churn patterns using interactive charts.
* Customer Risk Segmentation: Score customer records and group them into configurable risk categories.
* Model Performance Dashboard: Evaluate the classifier using accuracy, precision, recall, F1 score, ROC-AUC, and a confusion matrix.
* Feature Importance: Visualize the features the Random Forest model relies on most.
* Downloadable Reports: Export customer risk results as a CSV file.

Machine Learning

The project uses a Random Forest classifier trained on the IBM Telco Customer Churn dataset.

The preprocessing pipeline handles numerical and categorical features. Numerical features use median imputation and standardization, while categorical features use most-frequent imputation and one-hot encoding.

The model was evaluated using a held-out test set.

Technology Stack

* Python
* Pandas
* Scikit-learn
* Streamlit
* Plotly
* Matplotlib
* Joblib

Project Structure

ChurnGuardAI/
├── app.py
├── churnguard_model.pkl
├── Telco_customer_churn.xlsx
├── churnguard_test_data.csv
├── requirements.txt
├── README.md
└── .gitignore

The test-data file is optional for running the prediction and analytics features, but the model performance section requires it.

Run Locally

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd ChurnGuardAI

2. Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install -r requirements.txt

4. Run the application

python -m streamlit run app.py

Dataset

The application uses the IBM Telco Customer Churn dataset for historical customer analytics.

The dashboard’s historical analysis and its individual customer predictions are separate functions.

Limitations

* Predictions are estimates, not guarantees of customer behavior.
* Feature importance does not establish causation.
* Risk categories depend on configurable probability thresholds.
* Model performance may change when applied to different customer populations.
* Churn probabilities may not be perfectly calibrated.

Author

Developed as a machine learning and data analytics portfolio project.