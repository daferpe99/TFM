from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_model(input_dim, embedding_dim=32, lstm_units=16):
    """Construye y retorna el modelo."""
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=100),
        LSTM(lstm_units),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model