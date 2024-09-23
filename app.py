from flask import Flask, request, render_template
import numpy as np
from knn_model import train_knn_model, predict_disease
from dqn_model import train_dqn_model
import pandas as pd

app = Flask(__name__)

# Load KNN and DQN models
knn, mlb = train_knn_model()
dqn_agent = train_dqn_model()

# Load the dataset for displaying drug recommendations
df = pd.read_csv('symptoms_dataset_400_with_dosage.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_symptoms = request.form['symptoms'].split(', ')
    
    # Convert input symptoms into binary form using the same MultiLabelBinarizer used in KNN
    input_symptoms_bin = mlb.transform([input_symptoms])
    
    # Predict disease using KNN
    disease = predict_disease(knn, mlb, input_symptoms)
    
    # Fetch the drug recommendations based on the predicted disease
    drug_info = df[df['Disease'] == disease[0]].iloc[0].to_dict()
    
    # Convert input_symptoms_bin to a numpy array (state) for the DQN agent
    state = np.array(input_symptoms_bin)
    
    # Simulate reinforcement learning action based on user feedback
    action = dqn_agent.act(state)
    
    # Render the result page with prediction details
    return render_template('result.html', disease=disease[0], drug_info=drug_info, action=action)

if __name__ == '__main__':
    app.run(debug=True)
