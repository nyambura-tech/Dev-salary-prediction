'''
Simple flask web app

Run:
python app/flask_app.py
 then open browser:
    http://127.0.0.1:5000
'''
from pathlib import Path
import joblib 
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, jsonify


# App setup
app = Flask(__name__)
current_dir = Path(__file__).resolve().parent
print(current_dir)

 
MODEL_PATH = current_dir.parent/ 'models/salary_pipeline.pkl'
 
# picking the model
pipeline = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")
 
# html form
HTML = """
<html>
    <head><title> Salary predictor </title></head>
 
    <body>
        <h1> Developer salary predictor </h1>
 
        <form method="POST" action="/predict"> 
            <label> Country: </label> <br>
            <select name = "country">
                <option>United States of America</option>
                <option>Germany</option>
                <option>United Kingdom of Great Britain and Northern  Ireland</option>
                <opyion>Indian</option>
                <opyion>Canada</option>
                <opyion>France</option>
                <opyion>Brazil</option>
                <opyion>Others</option>
            </select>
             <br><br>

            <label> Education Level:</label>
            <select name="education">
                <option>Bachelor's</option>
                <option>Master's</option>
                <option>PhD</option>
                <option>Associate's</option>
                <option>Some College</option>
                <option>High School</option>
                </select>
                <br><br>


            <label> Employment type: </label>
            <select name="employment">
                <option>Full-time</option>
                <option>Part-time</option>
                <option>Student</option>
                <option>Freelance/Self-employed</option>
                <option>Other</option>
            </select>
            <br><br>


            <label> Years of experience </label>
            <input type="number" name="years" value="5" min="0" max="40">
            <br><br>


            <label>Number of programming languages used: </label>
            <input type="number" name="languages" value="4" min="1" max="20">
            <br><br>
         <button type="submit"> Predict salary</button>
        </form>
        {%if salary%}
            <hr>
            <h2> predicted salary: $ {{ salary }}</h2>
        {%endif%}
    </body>
</html>
"""
 
@app.route('/')
def home():
    return render_template_string(HTML)
 
@app.route("/predict", methods=['POST'])
def predict():
    input_df = pd.DataFrame([{
        'Country': request.form.get('country'),
        'YearsCode': float(request.form.get('years')),
        'EdLevel': request.form.get('education'),
        'Employment':request.form.get('employment'),
        'LanguageCount':float(request.form.get('languages'))

    }])
    prediction = pipeline.predict(input_df)[0]
    salary = f"{int(np.clip(prediction, 10_000, 500_000)):,}"
    return render_template_string(HTML, salary=salary)
 
if __name__ == '__main__':
    app.run(debug=True)