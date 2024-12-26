def evaluate_model(model, test_X, test_Y):
    """Evalúa el modelo en datos de prueba."""
    loss, accuracy = model.evaluate(test_X, test_Y)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy