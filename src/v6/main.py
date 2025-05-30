from data_load import load_data, split_data
import preprocess
from model import build_model, build_multimodal_modelv5, build_model_tuner
from training import cross_validate_model, train_model, cross_validate_with_best_hp
from evaluation import evaluate_model, evaluate_permutation_importance, save_feature_importance, save_results_completo, save_cv_results, save_tuner_results
from visualization import plot_samples, plot_training_history, plot_confusion_matrix, plot_correlation_matrix
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import os
import time
import keras_tuner as kt

# Cambiar entre 'completo' o 'reducido'
METADATA_VERSION = "reducido"
USE_KERAS_TUNER = True

# Configurar rutas de guardado según versión de metadatos
RESULTS_TAG = "completo" if METADATA_VERSION == "completo" else "reducido"
PATH_CV_RESULTS = f"../../resultados/v6_cross_validation_{RESULTS_TAG}.txt"
PATH_CV_RESULTS_KERASTUNER = f"../../resultados/v6_cross_validation_hpo_{RESULTS_TAG}.txt"
PATH_PREDICTIONS = f"../../resultados/v6_predicciones_finales_{RESULTS_TAG}.csv"
PATH_IMPORTANCE = f"../../resultados/v6_importancia_caracteristicas_{RESULTS_TAG}.txt"

# Establecer semillas para reproducibilidad
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- CARGA Y PREPROCESAMIENTO DE DATOS ---
start_time = time.time()
data = load_data()
plot_samples(data)
data = preprocess.downsampling_data(data)
plot_samples(data)
data["tweet_sample_clean"] = data["tweet_sample"].apply(preprocess.clean_tweet)
data["tweet_sample_clean"] = data["tweet_sample_clean"].apply(preprocess.normalize_text)
data["tweet_sample_clean"] = data["tweet_sample_clean"].apply(preprocess.remove_emojis)
data = data[data["tweet_sample_clean"].str.len() > 2]
data = data[data["tweet_sample_clean"].notna()]
data = preprocess.preprocess_user_metrics(data)
end_time = time.time()
print(f"Tiempo de carga y preprocesamiento (v4): {end_time - start_time:.2f} segundos")

x_text = data["tweet_sample_clean"].values

if METADATA_VERSION == "completo":
    selected_features = [
        "followers_count_log_scaled", "following_count_log_scaled", "tweet_count_log_scaled",
        "user_age_days_scaled", "followers_following_ratio_log_scaled", "tweets_per_day_log_scaled",
        "verified", "protected", "has_profile_image", "has_url"
    ]
else:
    selected_features = ["user_age_days_scaled", "verified", "has_url", "has_profile_image"]

x_numeric = data[selected_features].values
y = (data["label"] == "bot").astype(int).values

print(f"Tamaño de x_text: {len(x_text)}")
print(f"Tamaño de x_numeric: {len(x_numeric)}")
print(f"Tamaño de y: {len(y)}")

# Vectorizador global
temp_vectorizer = TextVectorization(output_sequence_length=150, max_tokens=10000)
temp_vectorizer.adapt(x_text)

# --- ENTRENAMIENTO CON KERAS TUNER ---
if USE_KERAS_TUNER:
    print("\n🚀 Ejecutando pipeline con Keras Tuner")

    def model_builder(hp):
        model, _ = build_model_tuner(hp, n_numeric_features=x_numeric.shape[1], text_vectorizer=temp_vectorizer)
        return model

    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='tuner_dir',
        project_name='twitter_bot_detection'
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1)

    x_train_text, x_val_text, x_train_num, x_val_num, y_train, y_val = train_test_split(
        x_text, x_numeric, y, test_size=0.2, random_state=42, stratify=y
    )

    tuner.search(
        {"text_input": x_train_text, "numeric_input": x_train_num},
        y_train,
        validation_data=(
            {"text_input": x_val_text, "numeric_input": x_val_num},
            y_val
        ),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    tuned_model = tuner.hypermodel.build(best_hp)

    checkpoint_cb = ModelCheckpoint(
        filepath="mejor_modelo",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    history = tuned_model.fit(
        {"text_input": x_train_text, "numeric_input": x_train_num},
        y_train,
        validation_data=({"text_input": x_val_text, "numeric_input": x_val_num}, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop, reduce_lr, checkpoint_cb],
        verbose=1
    )

    loss, acc = tuned_model.evaluate({"text_input": x_val_text, "numeric_input": x_val_num}, y_val, verbose=1)
    plot_training_history(history)

    # Guardar predicciones
    y_pred_probs = tuned_model.predict({"text_input": x_val_text, "numeric_input": x_val_num})
    y_pred = (y_pred_probs > 0.5).astype(int)

    pd.DataFrame({
        "true_label": y_val,
        "predicted_prob": y_pred_probs.flatten(),
        "predicted_label": y_pred.flatten()
    }).to_csv(PATH_PREDICTIONS, index=False)

    # Validación cruzada
    cv_metrics = cross_validate_with_best_hp(
        best_hp,
        x_text=x_text,
        x_numeric=x_numeric,
        y=y,
        n_splits=5,
        batch_size=32,
        epochs=10,
        text_vectorizer=temp_vectorizer
    )

    # Convertir los hiperparámetros en diccionario limpio
    hp_dict = {param: best_hp.get(param) for param in best_hp.values}

    save_cv_results(
        file_path=PATH_CV_RESULTS,
        model_name=f"Modelo final HPO {RESULTS_TAG} + validación cruzada",
        metrics_dict=cv_metrics,
        additional_info={
            "Tipo de validación": "StratifiedKFold",
            "Semilla": SEED,
            "Hiperparámetros óptimos": hp_dict
        }
    )

# --- ENTRENAMIENTO MANUAL SIN KERAS TUNER ---
else:
    print("\n🧪 Ejecutando pipeline sin Keras Tuner (modelo base)")
    metrics = cross_validate_model(
        build_model_fn=build_multimodal_modelv5,
        x_text=x_text,
        x_numeric=x_numeric,
        y=y,
        n_splits=5,
        batch_size=32,
        epochs=10
    )

    save_cv_results(
        file_path=PATH_CV_RESULTS,
        model_name=f"Modelo base {RESULTS_TAG} con validación cruzada",
        metrics_dict=metrics,
        additional_info={"Tipo de validación": "StratifiedKFold", "Semilla": SEED}
    )
