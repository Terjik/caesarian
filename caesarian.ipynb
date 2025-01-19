import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 1: Data Preprocessing
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Handle missing values using KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Normalize features using Min-Max scaling
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)

    return data_scaled

def feature_selection(X_train, y_train):
    # Use Recursive Feature Elimination (RFE) with a RandomForestClassifier
    estimator = RandomForestClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    selector.fit(X_train, y_train)
    return selector

def split_data(data, train_size=0.7, validate_size=0.15):
    X = data.drop(columns=['Caesarian'])
    y = data['Caesarian']

    # Split into training and temp (validation+test) datasets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

    # Calculate correct validation size relative to remaining data
    validate_ratio = validate_size / (1 - train_size)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=validate_ratio, random_state=42, stratify=y_temp)

    return (X_train, y_train), (X_validate, y_validate), (X_test, y_test)

# Step 2: Initialize Base Classifiers with Hyperparameter Tuning
def get_tuned_classifier(clf, param_grid, X, y):
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Load data
data = load_data("https://github.com/Terjik/caesarian/raw/refs/heads/main/caesarian_converted.csv")
data = preprocess_data(data)

# Split data
(train_X, train_y), (validate_X, validate_y), (test_X, test_y) = split_data(data)

# Perform feature selection only on training data
selector = feature_selection(train_X, train_y)
selected_features = train_X.columns[selector.support_]

# Apply feature selection to all datasets
train_X, validate_X, test_X = train_X[selected_features], validate_X[selected_features], test_X[selected_features]

# Define parameter grids
param_grid_gbc = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
param_grid_bc = {'n_estimators': [10, 50]}
param_grid_etc = {'n_estimators': [50, 100], 'max_features': ['auto', 'sqrt']}
param_grid_svm = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}

# Initialize and tune classifiers
gbc = get_tuned_classifier(GradientBoostingClassifier(random_state=42), param_grid_gbc, train_X, train_y)
rf = get_tuned_classifier(RandomForestClassifier(random_state=42), param_grid_rf, train_X, train_y)
bc = get_tuned_classifier(BaggingClassifier(random_state=42), param_grid_bc, train_X, train_y)
etc = get_tuned_classifier(ExtraTreesClassifier(random_state=42), param_grid_etc, train_X, train_y)
svm = get_tuned_classifier(SVC(probability=True, random_state=42), param_grid_svm, train_X, train_y)

# Train Base Classifiers
base_classifiers = [gbc, rf, bc, etc, svm]
for clf in base_classifiers:
    clf.fit(train_X, train_y)

# Step 3: Create Voting Classifier with Weighted Voting
voting_classifier = VotingClassifier(
    estimators=[
        ('GradientBoosting', gbc),
        ('RandomForest', rf),
        ('Bagging', bc),
        ('ExtraTrees', etc),
        ('SVM', svm)
    ],
    voting='soft',
    weights=[2, 3, 1, 1, 2]  # Example weights based on validation performance
)
voting_classifier.fit(train_X, train_y)

# Step 4: Evaluate Models
def evaluate_model(clf, X, y):
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else None

    metrics = {
        'Accuracy': accuracy_score(y, predictions),
        'Precision': precision_score(y, predictions),
        'Recall': recall_score(y, predictions),
        'F1-Score': f1_score(y, predictions),
    }
    try:
        metrics['AUC'] = roc_auc_score(y, probabilities) if probabilities is not None else 'N/A'
    except:
        metrics['AUC'] = 'N/A'  # Handles cases where only one class is predicted
    
    return metrics

models = {'GBC': gbc, 'RF': rf, 'BC': bc, 'ETC': etc, 'SVM': svm, 'VotingClassifier': voting_classifier}
for name, model in models.items():
    print(f"{name} Performance:", evaluate_model(model, test_X, test_y))

# Step 5: Interpret Results
print("Feature Importance:", rf.feature_importances_)
