import time
import io
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

def evaluate_model(model, test_X, test_Y):
    """Evalúa el modelo en datos de prueba."""
    loss, accuracy = model.evaluate(test_X, test_Y)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy

def evaluate_model(model, x_text_test, x_num_test, y_test):
    loss, accuracy = model.evaluate(
        {"text_input": x_text_test, "numeric_input": x_num_test}, y_test, verbose=1
    )
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict({"text_input": x_text_test, "numeric_input": x_num_test})
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return loss, accuracy


def evaluate_permutation_importance(model, x_text, x_numeric, y, text_vectorizer, feature_names, scoring=f1_score):
    from sklearn.base import BaseEstimator, ClassifierMixin

    class KerasWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model, x_text):
            self.model = model
            self.x_text = x_text
            self.classes_ = np.array([0, 1])  # AÑADIDO: requerimiento de sklearn

        def fit(self, X, y):
            return self  # Ya está entrenado

        def predict(self, X):
            # Solo X numérico, el texto es constante
            predictions = self.model.predict({"text_input": self.x_text, "numeric_input": X})
            return (predictions > 0.5).astype(int)

    wrapper = KerasWrapper(model, x_text)

    print("🔍 Evaluación de importancia de características (Permutation Importance)")
    result = permutation_importance(
        estimator=wrapper,
        X=x_numeric,
        y=y,
        scoring='f1',
        n_repeats=10,
        random_state=42
    )

    # Mostrar resultados
    importance_scores = dict(zip(feature_names, result.importances_mean))
    for feat, score in importance_scores.items():
        print(f"{feat}: {score:.4f}")

    return importance_scores

def save_results(file_path, model, loss, accuracy, model_name, additional_info=None):
    with open(file_path, 'a') as file:
        # Guardar el nombre del modelo
        file.write(f"Modelo: {model_name}\n")

        # Capturar el resumen del modelo
        file.write("Resumen del modelo:\n")
        summary_stream = io.StringIO()
        model.summary(print_fn=lambda x: summary_stream.write(x + "\n"))
        file.write(summary_stream.getvalue())

        # Guardar resultados de evaluación
        file.write(f"\nResultados de evaluación:\n")
        file.write(f"Pérdida: {loss:.4f}\n")
        file.write(f"Precisión: {accuracy:.4f}\n")
        
        # Guardar información adicional si existe
        if additional_info:
            file.write("\nInformación adicional:\n")
            for key, value in additional_info.items():
                file.write(f"{key}: {value}\n")
        
        file.write("-" * 50 + "\n")

def save_results_completo(file_path, model, loss, accuracy, model_name, x_text_test, x_num_test, y_test, additional_info=None):
    # Obtener predicciones
    y_pred_probs = model.predict({"text_input": x_text_test, "numeric_input": x_num_test})
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    with open(file_path, 'a', encoding='utf-8') as file:
        # Guardar nombre del modelo
        file.write(f"Modelo: {model_name}\n")

        # Guardar resumen del modelo
        file.write("Resumen del modelo:\n")
        summary_stream = io.StringIO()
        model.summary(print_fn=lambda x: summary_stream.write(x + "\n"))
        file.write(summary_stream.getvalue())

        # Resultados de evaluación
        file.write("\nResultados de evaluación:\n")
        file.write(f"Pérdida: {loss:.4f}\n")
        file.write(f"Precisión (accuracy): {accuracy:.4f}\n")

        # Guardar métricas adicionales
        file.write("\n📊 Reporte de clasificación:\n")
        file.write(report + "\n")

        file.write("🧩 Matriz de confusión:\n")
        file.write(np.array2string(cm, separator=', ') + "\n")

        # Información adicional opcional
        if additional_info:
            file.write("\nInformación adicional:\n")
            for key, value in additional_info.items():
                file.write(f"{key}: {value}\n")

        file.write("-" * 60 + "\n")

def save_cv_results(file_path, model_name, metrics_dict, additional_info=None):
    from datetime import datetime
    with open(file_path, 'a') as file:
        file.write(f"Modelo: {model_name}\n")
        file.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for metric, values in metrics_dict.items():
            mean = np.mean(values)
            std = np.std(values)
            file.write(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}\n")
        
        if additional_info:
            file.write("\nInformación adicional:\n")
            for key, value in additional_info.items():
                file.write(f"{key}: {value}\n")
        
        file.write("-" * 60 + "\n")

def save_feature_importance(file_path, importance_scores, model_name, additional_info=None):
    with open(file_path, 'a') as file:
        file.write(f"Modelo: {model_name}\n")
        file.write("Importancia de características (Permutation Importance):\n")
        for feature, score in importance_scores.items():
            file.write(f"{feature}: {score:.4f}\n")
        if additional_info:
            file.write("\nInformación adicional:\n")
            for key, value in additional_info.items():
                file.write(f"{key}: {value}\n")
        file.write("-" * 50 + "\n")



def save_tuner_results(tuner, file_path, metadata_version="completo"):
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    results = {
        "best_hyperparameters": {param: best_hp.get(param) for param in best_hp.values},
        "metadata_version": metadata_version,
        "project_name": tuner.project_name,
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Resultados del tuner guardados en: {file_path}")