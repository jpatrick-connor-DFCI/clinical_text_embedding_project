import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/VTE_data/processed_datasets/")

full_IO_prediction_dataset = pd.read_csv(os.path.join(DATA_PATH, 'treatment_prediction/IO_first_line_preds.csv'))

X = full_IO_prediction_dataset[[col for col in full_IO_prediction_dataset.columns if ('IMAGING' in col) or ('PATHOLOGY' in col) or ('CLINICIAN' in col)]]
y = full_IO_prediction_dataset[['IO_status']].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Fit logistic regression
clf = LogisticRegression(max_iter=1000, solver="lbfgs")  # lbfgs works well for most cases
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))