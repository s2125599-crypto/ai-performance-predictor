from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'student_performance_demo.csv'
MODEL = BASE / 'model.joblib'
FEATURES = [
    'attendance_pct', 'internal_marks_30', 'assignment_marks_50',
    'study_hours_per_day', 'previous_score_pct', 'backlogs', 'absences'
]

df = pd.read_csv(DATA)
X = df[FEATURES]
y = df['performance']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'
    ),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', probability=True, random_state=42))
    ])
}
rows = []
trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average='weighted', zero_division=0
    )
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': p,
        'Recall': r,
        'F1': f1
    })
    trained[name] = model

results = pd.DataFrame(rows).sort_values('F1', ascending=False)
results.to_csv(BASE / 'data' / 'model_comparison_runtime.csv', index=False)
best_name = results.iloc[0]['Model']
joblib.dump(trained[best_name], MODEL)
print(results.to_string(index=False))
print(f'\nSaved best model: {best_name}')
