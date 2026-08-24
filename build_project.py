from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE/'data'
FIG = BASE/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(2026)
n = 800
attendance = np.clip(rng.normal(79, 11, n), 45, 100)
internal = np.clip(rng.normal(21, 4.0, n), 8, 30)
assignment = np.clip(rng.normal(39, 7.0, n), 10, 50)
study_hours = np.clip(rng.normal(3.5, 1.5, n), 0.5, 9)
previous = np.clip(rng.normal(66, 15, n), 25, 95)
backlogs = np.clip(rng.poisson(0.55, n), 0, 4)
absences = np.clip(np.round((100-attendance)/2.4 + rng.normal(0,2,n)), 0, 25).astype(int)

latent = (
    0.32*attendance +
    1.10*internal +
    0.48*assignment +
    3.2*study_hours +
    0.40*previous -
    5.5*backlogs -
    0.45*absences +
    rng.normal(0, 7.0, n)
)
latent = (latent - latent.min())/(latent.max()-latent.min())*100

def label(v):
    if v < 45: return 'Low'
    if v < 62: return 'Average'
    if v < 80: return 'Good'
    return 'Excellent'

df = pd.DataFrame({
    'attendance_pct': np.round(attendance,1),
    'internal_marks_30': np.round(internal,1),
    'assignment_marks_50': np.round(assignment,1),
    'study_hours_per_day': np.round(study_hours,1),
    'previous_score_pct': np.round(previous,1),
    'backlogs': backlogs.astype(int),
    'absences': absences,
})
df['performance'] = [label(v) for v in latent]
df.to_csv(DATA/'student_performance_demo.csv', index=False)

features = list(df.columns[:-1])
X = df[features]; y = df['performance']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

models = {
 'Logistic Regression': Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000))]),
 'Decision Tree': DecisionTreeClassifier(max_depth=6,random_state=42),
 'Random Forest': RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,class_weight='balanced'),
 'SVM': Pipeline([('scaler',StandardScaler()),('model',SVC(kernel='rbf',probability=True,random_state=42))]),
}
results=[]
trained={}
labels=['Low','Average','Good','Excellent']
for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    p,r,f,_=precision_recall_fscore_support(y_test,pred,average='weighted',zero_division=0)
    results.append([name,accuracy_score(y_test,pred),p,r,f])
    trained[name]=model
res=pd.DataFrame(results,columns=['Model','Accuracy','Precision','Recall','F1'])
res.to_csv(DATA/'model_comparison.csv',index=False)

best_name=res.sort_values('F1',ascending=False).iloc[0]['Model']
best=trained[best_name]
joblib.dump(best, BASE/'model.joblib')

pred=best.predict(X_test)
cm=confusion_matrix(y_test,pred,labels=labels)
report=classification_report(y_test,pred,labels=labels,zero_division=0)
(Path(BASE/'evaluation.txt')).write_text(
    f'Best model: {best_name}\n'
    f'Test samples: {len(y_test)}\n\n{res.to_string(index=False)}\n\nClassification report:\n{report}\n\nConfusion matrix:\n{cm}\n', encoding='utf-8')

# Feature importance if available
if hasattr(best,'feature_importances_'):
    imp=best.feature_importances_
else:
    imp=np.zeros(len(features))
    # approximate ranking for linear SVC / LR pipeline is intentionally omitted
fi=pd.DataFrame({'Feature':features,'Importance':imp}).sort_values('Importance',ascending=False)
fi.to_csv(DATA/'feature_importance.csv',index=False)

# 1. class distribution
plt.figure(figsize=(8,5))
df['performance'].value_counts().reindex(labels).plot(kind='bar')
plt.title('Performance Class Distribution (Demonstration Dataset)')
plt.xlabel('Performance Category'); plt.ylabel('Number of Students')
plt.tight_layout(); plt.savefig(FIG/'class_distribution.png',dpi=180); plt.close()

# 2. model comparison
plt.figure(figsize=(8,5))
plt.bar(res['Model'],res['Accuracy'])
plt.ylim(0,1); plt.title('Model Accuracy Comparison')
plt.ylabel('Hold-out Test Accuracy'); plt.xticks(rotation=20,ha='right')
for i,v in enumerate(res['Accuracy']): plt.text(i,v+0.02,f'{v:.2f}',ha='center')
plt.tight_layout(); plt.savefig(FIG/'model_accuracy.png',dpi=180); plt.close()

# 3. confusion matrix
plt.figure(figsize=(6.5,5.5))
plt.imshow(cm)
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.xticks(range(4),labels,rotation=25,ha='right'); plt.yticks(range(4),labels)
for i in range(4):
    for j in range(4): plt.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.colorbar(); plt.tight_layout(); plt.savefig(FIG/'confusion_matrix.png',dpi=180); plt.close()

# 4. attendance vs performance summary
summary=df.groupby('performance',sort=False)['attendance_pct'].mean().reindex(labels)
plt.figure(figsize=(8,5)); summary.plot(kind='bar')
plt.title('Average Attendance by Performance Category'); plt.ylabel('Attendance (%)'); plt.xlabel('Performance Category')
plt.tight_layout(); plt.savefig(FIG/'attendance_by_category.png',dpi=180); plt.close()

# Metadata
meta={
 'samples': int(len(df)), 'features': features, 'best_model': best_name,
 'test_samples': int(len(y_test)), 'accuracy': float(res.loc[res['Model']==best_name,'Accuracy'].iloc[0]),
 'precision': float(res.loc[res['Model']==best_name,'Precision'].iloc[0]),
 'recall': float(res.loc[res['Model']==best_name,'Recall'].iloc[0]),
 'f1': float(res.loc[res['Model']==best_name,'F1'].iloc[0]),
}
import json
(BASE/'evaluation.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
