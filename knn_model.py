import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def train_knn_model():
    df = pd.read_csv('symptoms_dataset_400_with_dosage.csv')
    symptoms = df['Symptoms'].apply(lambda x: x.split(', '))
    diseases = df['Disease']
    
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(symptoms)
    y = diseases

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    return knn, mlb
def predict_disease(knn, mlb, symptoms):
    symptoms_encoded = mlb.transform([symptoms])
    prediction = knn.predict(symptoms_encoded)
    return prediction
