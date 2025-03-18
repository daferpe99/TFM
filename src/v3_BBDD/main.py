from data_load import load_data, split_data
import preprocess
from model import build_model, build_multimodal_model
from training import train_model
from evaluation import evaluate_model, save_results
from visualization import plot_samples, plot_training_history
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np



# Carga y preprocesa los datos
data = load_data()
print(data.head())
plot_samples(data)


#Donwsampling de los datos
#data = preprocess.downsampling_data(data)
#plot_samples(data)

# Preprocesar texto de los tweets
data["tweet_sample_clean"] = data["tweet_sample"].apply(preprocess.clean_tweet)

# Filtrar datos inválidos
data = data[data["tweet_sample_clean"].str.len() > 2]
data = data[data["tweet_sample_clean"].notna()]

# Procesar fecha de creación del usuario
# Convertir a datetime sin zona horaria
data["user_creation"] = pd.to_datetime(data["user_creation"], utc=True).dt.tz_localize(None)
#data["last_tweet"] = pd.to_datetime(data["last_tweet"], utc=True).dt.tz_localize(None)

# Calcular métricas adicionales a nivel de usuario
data["user_age_days"] = (pd.Timestamp.now().normalize() - data["user_creation"]).dt.days
#data["last_tweet_age_days"] = (pd.Timestamp.now().normalize() - data["last_tweet"]).dt.days

# Escalar las métricas
data["user_age_scaled"] = data["user_age_days"] / data["user_age_days"].max()

# Separar características y etiquetas
x_text = data["tweet_sample_clean"].values
x_numeric = data["user_age_days"].values
y = (data["label"] == "bot").astype(int).values


# Verificar tamaños antes de train_test_split
print(f"Tamaño de x_text: {len(x_text)}")
print(f"Tamaño de x_numeric: {len(x_numeric)}")
print(f"Tamaño de y: {len(y)}")

# Dividir datos en entrenamiento y prueba

x_text_train, x_text_test, x_numeric_train, x_numeric_test, y_train, y_test = train_test_split(
    x_text, x_numeric, y, test_size=0.2, random_state=42
)

# Construir y entrenar el modelo
model, text_vectorizer = build_multimodal_model()

#Ajustar vectorizador del texto
text_vectorizer.adapt(x_text_train)


#Entramiento del modelo
history = train_model(model, x_text_train, x_text_test, x_numeric_train, x_numeric_test, y_train, y_test)
#Visualizar resultados
plot_training_history(history)

# Evaluación
loss, accuracy = evaluate_model(model, x_text_test, x_numeric_test, y_test)


#save_results("../../resultados/v3_texto+fechacreacionusuario_downsampled.txt",model=model, loss=loss, accuracy=accuracy, model_name="Modelo LSTM multimodal texto + fecha de creacion usuario con downsampling")

