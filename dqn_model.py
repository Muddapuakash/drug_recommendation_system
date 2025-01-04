import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        return random.choice(range(self.action_size))

def train_dqn_model():
    df = pd.read_csv('symptoms_dataset_400_with_dosage.csv')
    state_size = len(df['Symptoms'][0].split(', '))
    action_size = 3  # Example, assuming 3 possible actions
    
    agent = DQNAgent(state_size, action_size)
    return agent
