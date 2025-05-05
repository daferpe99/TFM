import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Input, TextVectorization, Concatenate, Dense, Flatten, Lambda, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_model(input_dim, embedding_dim=32, lstm_units=16):
    """Construye y retorna el modelo."""
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=100),
        LSTM(lstm_units),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_modelv3(vocab_size=10000, embedding_dim=32, lstm_units=16):
     # Entrada de texto
    text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
    # Procesamiento de texto
    text_vectorizer = TextVectorization(vocab_size, output_sequence_length=100)
    text_vectorized = text_vectorizer(text_input)
    text_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100)(text_vectorized)
    text_output = LSTM(lstm_units)(text_embeddings)

    # Entrada numérica
    num_input = Input(shape=(1,), dtype=tf.float32, name='numeric_input')

    # Concatenar ambas entradas
    combined = Concatenate()([text_output, num_input])

    # Capas densas
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    # Crear el modelo
    model = Model(inputs=[text_input, num_input], outputs=output)

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, text_vectorizer

def build_multimodal_modelv4(vocab_size=10000, embedding_dim=32, lstm_units=16):
    # Entrada de texto
    text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
    text_vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=150)
    text_vectorized = text_vectorizer(text_input)

    # Embedding + LSTM bidireccional
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_vectorized)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(lstm_units // 2))(x)  # Capa adicional
    x = Dropout(0.3)(x)

    # Entrada numérica
    num_input = Input(shape=(1,), dtype=tf.float32, name='numeric_input')

    # Capa de fusión
    combined = Concatenate()([x, num_input])

    # Capas densas adicionales
    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)

    # Capa de salida
    output = Dense(1, activation='sigmoid')(x)

    # Modelo final
    model = Model(inputs=[text_input, num_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, text_vectorizer

def build_multimodal_modelv5(vocab_size=10000, embedding_dim=32, lstm_units=16, n_numeric_features=10):
    text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
    text_vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=150)
    text_vectorized = text_vectorizer(text_input)
    
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_vectorized)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(lstm_units // 2))(x)
    x = Dropout(0.3)(x)

    # Entrada de metadatos
    num_input = Input(shape=(n_numeric_features,), dtype=tf.float32, name='numeric_input')
    combined = Concatenate()([x, num_input])

    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[text_input, num_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, text_vectorizer