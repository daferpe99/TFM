import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import resample

# Cargar label.csv directamente en un DataFrame
label_df = pd.read_csv('label.csv', header=0, names=["label_id", "label_value"])

# Cargar tweet.json y convertirlo en un DataFrame
with open('tweet.json', mode='r') as tweet_file:
    tweets = json.load(tweet_file)
tweet_df = pd.DataFrame(tweets)

# Crear la columna 'label_id' en tweet_df para que coincida con el formato de 'label_df'
tweet_df['label_id'] = 'u' + tweet_df['author_id'].astype(str)

# Hacer un merge entre tweet_df y label_df usando 'label_id' como clave
merged_df = tweet_df.merge(label_df, on='label_id', how='left')

# Seleccionar solo las columnas relevantes: 'label_value' y 'text'
result_df = merged_df[['label_value', 'text']].rename(columns={'label_value': 'ordered_label', 'text': 'tweet_sample'})

# Mostrar el DataFrame resultante
print(result_df.head())


sns.countplot(x='ordered_label', data=result_df)
plt.xlabel("Etiquetas")
plt.ylabel("Cantidad")
plt.show()


# DOWNSAMPLING
# Separar las clases en dos DataFrames: clase mayoritaria y clase minoritaria
majority_class = result_df[result_df['ordered_label'] == 'human']
minority_class = result_df[result_df['ordered_label'] == 'bot']

# Aplicar downsampling a la clase mayoritaria
majority_downsampled = resample(
    majority_class,
    replace=False,                 # Sin reemplazo
    n_samples=len(minority_class), # Igualar el número de muestras de la clase minoritaria
    random_state=42                # Para reproducibilidad
)

# Combinar el downsampled de la clase mayoritaria con la clase minoritaria
downsampled_data = pd.concat([majority_downsampled, minority_class])

# Opcional: mezclar el dataset para eliminar cualquier patrón
downsampled_data = downsampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Mostrar el conteo de etiquetas después del downsampling
print(downsampled_data['ordered_label'].value_counts())

sns.countplot(x='ordered_label', data=downsampled_data)
plt.xlabel("Etiquetas")
plt.ylabel("Cantidad")
plt.show()

# Convertir 'ordered_label' en binario: 'human' -> 0 y 'bot' -> 1
downsampled_data['ordered_label'] = downsampled_data['ordered_label'].map({'human': 0, 'bot': 1})

# train-test split
train_X, test_X, train_Y, test_Y = train_test_split(downsampled_data['tweet_sample'],
                                                    downsampled_data['ordered_label'],
                                                    test_size=0.2,
                                                    random_state=42)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
 
# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)
 
# Pad sequences to have the same length
max_len = 100  # maximum sequence length
train_sequences = pad_sequences(train_sequences,
                                maxlen=max_len, 
                                padding='post', 
                                truncating='post')
test_sequences = pad_sequences(test_sequences, 
                               maxlen=max_len, 
                               padding='post', 
                               truncating='post')

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                    output_dim=32, 
                                    input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
 
# Print the model summary
model.summary()


model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = ['accuracy'],
              optimizer = 'adam')

es = EarlyStopping(patience=3,
                   monitor = 'val_accuracy',
                   restore_best_weights = True)
 
lr = ReduceLROnPlateau(patience = 2,
                       monitor = 'val_loss',
                       factor = 0.5,
                       verbose = 0)


# Train the model
history = model.fit(train_sequences, train_Y,
                    validation_data=(test_sequences, test_Y),
                    epochs=20, 
                    batch_size=32,
                    callbacks = [lr, es]
                   )


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss :',test_loss)
print('Test Accuracy :',test_accuracy)