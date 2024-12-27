import matplotlib.pyplot as plt
import seaborn as sns

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
    sns.countplot(x='ordered_label', data=data)
    plt.xlabel("Etiquetas")
    plt.ylabel("Cantidad")
    plt.show()
