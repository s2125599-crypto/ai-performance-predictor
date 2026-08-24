from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
model = joblib.load(BASE / 'model.joblib')
FEATURES = [
    'attendance_pct', 'internal_marks_30', 'assignment_marks_50',
    'study_hours_per_day', 'previous_score_pct', 'backlogs', 'absences'
]
LABELS = ['Low', 'Average', 'Good', 'Excellent']


def get_suggestions(v):
    suggestions = []
    if v['attendance_pct'] < 75:
        suggestions.append('Raise attendance toward 75% or higher where possible.')
    if v['internal_marks_30'] < 18:
        suggestions.append('Focus on internal-test preparation and revision of weak topics.')
    if v['assignment_marks_50'] < 35:
        suggestions.append('Complete assignments on time and use feedback to improve.')
    if v['study_hours_per_day'] < 2:
        suggestions.append('Increase focused study time gradually and maintain a weekly schedule.')
    if v['previous_score_pct'] < 50:
        suggestions.append('Revisit fundamentals from previous units before advanced topics.')
    if v['backlogs'] > 0:
        suggestions.append('Prioritize backlog subjects with a separate weekly study plan.')
    if not suggestions:
        suggestions.append('Maintain the current routine and keep monitoring academic indicators.')
    return suggestions


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    probability = None
    suggestions = []
    error = None
    if request.method == 'POST':
        try:
            values = {
                'attendance_pct': float(request.form['attendance_pct']),
                'internal_marks_30': float(request.form['internal_marks_30']),
                'assignment_marks_50': float(request.form['assignment_marks_50']),
                'study_hours_per_day': float(request.form['study_hours_per_day']),
                'previous_score_pct': float(request.form['previous_score_pct']),
                'backlogs': int(request.form['backlogs']),
                'absences': int(request.form['absences']),
            }
            if not 0 <= values['attendance_pct'] <= 100:
                raise ValueError('Attendance must be between 0 and 100.')
            if not 0 <= values['internal_marks_30'] <= 30:
                raise ValueError('Internal marks must be between 0 and 30.')
            if not 0 <= values['assignment_marks_50'] <= 50:
                raise ValueError('Assignment marks must be between 0 and 50.')
            if not 0 <= values['study_hours_per_day'] <= 24:
                raise ValueError('Study hours must be between 0 and 24.')
            if not 0 <= values['previous_score_pct'] <= 100:
                raise ValueError('Previous score must be between 0 and 100.')
            if values['backlogs'] < 0 or values['absences'] < 0:
                raise ValueError('Backlogs and absences cannot be negative.')

            frame = pd.DataFrame([values])[FEATURES]
            result = model.predict(frame)[0]
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(frame)[0]
                probability = round(float(max(probs)) * 100, 1)
            suggestions = get_suggestions(values)
        except (ValueError, KeyError) as exc:
            error = str(exc)
    return render_template(
        'index.html', result=result, probability=probability,
        suggestions=suggestions, error=error
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
