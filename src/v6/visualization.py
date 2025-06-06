import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    """Genera gráficos de pérdida y precisión durante el entrenamiento."""
    # Pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Evolución de la pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()
    plt.show()

    # Precisión
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Evolución de la precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()
    plt.show()

def plot_samples(data):
    sns.countplot(x='label', data=data)
    plt.xlabel("Etiquetas")
    plt.ylabel("Cantidad")
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de confusión')
    plt.show()


def plot_correlation_matrix(data):
    numeric_features = data[[
    "followers_count_log_scaled",
    "following_count_log_scaled",
    "tweet_count_log_scaled",
    "user_age_days_scaled",
    "followers_following_ratio_log_scaled",
    "tweets_per_day_log_scaled",
    "verified",
    "protected",
    "has_profile_image",
    "has_url"
]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlación entre metadatos")
    plt.show()