from data_load import load_data, split_data
import preprocess
from model import build_model, build_multimodal_model
from training import train_model
from evaluation import evaluate_model, save_results
from visualization import plot_samples, plot_training_history, plot_confusion_matrix
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



# Carga y preprocesa los datos
data = load_data()
print(data.head())
plot_samples(data)


#Donwsampling de los datos
data = preprocess.downsampling_data(data)
plot_samples(data)

# Preprocesar texto de los tweets
data["tweet_sample_clean"] = data["tweet_sample"].apply(preprocess.clean_tweet)
# Normalizar el texto para prevenir errores de codificación
data["tweet_sample_clean"] = data["tweet_sample_clean"].apply(preprocess.normalize_text)
data["tweet_sample_clean"] = data["tweet_sample_clean"].apply(preprocess.remove_emojis)

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
    x_text, x_numeric, y, test_size=0.2, stratify=y, random_state=42,
)

#Convertir todo a arrays válidos
x_text_train = np.array(x_text_train).astype(str)
x_text_test = np.array(x_text_test).astype(str)
x_numeric_train = np.array(x_numeric_train).astype(np.float32)
x_numeric_test = np.array(x_numeric_test).astype(np.float32)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Calcular pesos de clase para balancear la penalización
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Pesos de clase:", class_weight_dict)

# Construir y entrenar el modelo
model, text_vectorizer = build_multimodal_model()


#Ajustar vectorizador del texto
text_vectorizer.adapt(x_text_train)

for i, text in enumerate(data["tweet_sample_clean"]):
    try:
        text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"❌ Error en fila {i}: {text}")


#Entramiento del modelo
history = train_model(model, x_text_train, x_text_test, x_numeric_train, x_numeric_test, y_train, y_test, class_weight=class_weight_dict)
#Visualizar resultados
plot_training_history(history)


print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))
# Evaluación
loss, accuracy = evaluate_model(model, x_text_test, x_numeric_test, y_test)

y_pred = (model.predict({"text_input": x_text_test, "numeric_input": x_numeric_test}) > 0.5).astype(int)
plot_confusion_matrix(y_test, y_pred)

#save_results("../../resultados/v4_texto+fechacreacionusuario_BBD_tweets1string&metricas.txt",model=model, loss=loss, accuracy=accuracy, model_name="Modelo LSTM multimodal texto + fecha de creacion usuario + carga de BBDD + agrupación de tweets en 1 string + métricas")

