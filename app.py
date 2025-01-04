from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from knn_model import train_knn_model, predict_disease
from dqn_model import train_dqn_model
import sqlite3

app = Flask(__name__)

# Load models and dataset
knn, mlb = train_knn_model()
dqn_agent = train_dqn_model()
df = pd.read_csv('symptoms_dataset_400_with_dosage.csv')

# Database setup for feedback
conn = sqlite3.connect('feedback.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS Feedback
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback TEXT NOT NULL);''')
conn.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms_input = request.form['symptoms'].split(',')
    disease = predict_disease(knn, mlb, symptoms_input)
    
    if disease:
        disease_name = disease[0]
        drug_info = df[df['Disease'] == disease_name].iloc[0].to_dict()
        state = np.array(mlb.transform([symptoms_input])) 
        action = dqn_agent.act(state)
        return render_template('result.html', disease=disease_name, drug_info=drug_info, action=action)
    else:
        return render_template('result.html', error="No matching disease found.")

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    with sqlite3.connect('feedback.db') as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO Feedback (feedback) VALUES (?)", (feedback,))
        conn.commit()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
