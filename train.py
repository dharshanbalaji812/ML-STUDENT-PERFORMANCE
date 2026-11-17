import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
data = pd.read_csv('Dataset/student.csv')
print("Original Dataset:")
print(data)
print("\nMedian Values:")
print("Study Hours:", data['study_hours'].median())
print("Attendance:", data['attendance'].median())
print("Previous Marks:", data['previous_marks'].median())
X = data[['study_hours', 'attendance', 'previous_marks']]
y = data['result']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n========== MODEL EVALUATION ==========")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average='weighted', zero_division=0
)
recall = recall_score(
    y_test, y_pred, average='weighted', zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average='weighted', zero_division=0
)
print("\nAccuracy:", accuracy * 100, "%")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))
if len(model.classes_) == 2:
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    print("Specificity:", specificity)
    y_probability = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probability)
    print("ROC-AUC Score:", auc)
print("\n========== EVALUATION COMPLETED ==========")
model.fit(X, y)
with open('models/student_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("\nPickle model saved successfully!")