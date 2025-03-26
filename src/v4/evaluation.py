import io
from sklearn.metrics import classification_report, confusion_matrix

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