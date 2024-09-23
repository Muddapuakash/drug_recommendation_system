import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Load dataset
def load_data():
    df = pd.read_csv('symptoms_dataset_400_with_dosage.csv')
    df['Symptoms'] = df['Symptoms'].apply(lambda x: x.split(', '))  # Convert string to list of symptoms
    return df

# Train the KNN model
def train_knn_model():
    df = load_data()
    
    # Convert symptoms to binary features using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['Symptoms'])
    
    # Use diseases as target
    y = df['Disease']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    return knn, mlb

# Predict based on symptoms
def predict_disease(knn, mlb, input_symptoms):
    input_symptoms_bin = mlb.transform([input_symptoms])
    return knn.predict(input_symptoms_bin)
