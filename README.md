# Enhancing Recommendation System

This web application provides a platform for predicting diseases based on symptoms, recommending drugs, and collecting user feedback. It uses machine learning models like K-Nearest Neighbors (KNN) for disease prediction and Deep Q-Networks (DQN) for personalized drug recommendations. The system integrates with a user-friendly Flask web interface and stores feedback in an SQLite database.

## Features

- **Disease Prediction**: Predict diseases based on symptoms using a KNN model.
- **Drug Recommendations**: Personalized drug suggestions through a DQN agent.
- **User Feedback**: Collect feedback from users and store it in a local SQLite database.
- **Web Interface**: Built with Flask, offering various pages like Home, About, Help, and Chat.
- **Model Integration**: KNN for disease classification and DQN for reinforcement learning-based drug recommendations.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Muddapuakash/drug_recommendation_system]
   cd drug_recommendation_system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download or create the dataset `symptoms_dataset_400_with_dosage.csv` and place it in the project root directory.

4. Run the Flask application:
   ```bash
   python app.py
   ```

The app will be accessible at `http://127.0.0.1:5000/`.

## File Structure

* `app.py`: Main Flask application file.
* `knn_model.py`: Script for training and using the KNN model for disease prediction.
* `dqn_model.py`: Script for training and using the DQN agent for drug recommendations.
* `symptoms_dataset_400_with_dosage.csv`: Dataset containing symptoms, diseases, and drug recommendations.
* `templates/`: Folder containing HTML templates for rendering web pages.
* `feedback.db`: SQLite database to store user feedback.
* `requirements.txt`: List of Python packages required to run the application.

## Usage

1. Navigate to the homepage and input symptoms in the provided form.
2. The system will predict the disease based on the input symptoms and provide personalized drug recommendations.
3. Users can submit feedback, which will be stored in the database for further improvements.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make changes, and create a pull request. Please ensure that your contributions align with the project's goal of improving the disease prediction and drug recommendation system.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* The KNN model for disease prediction is based on scikit-learn's KNeighborsClassifier.
* The DQN agent is based on reinforcement learning principles.
* The project uses Flask for the web interface and SQLite for database management.
