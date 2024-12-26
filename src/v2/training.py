from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model, train_X, train_Y, test_X, test_Y, batch_size=32, epochs=20):
    """Entrena el modelo y retorna el historial."""
    es = EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy')
    lr = ReduceLROnPlateau(patience=2, factor=0.5, monitor='val_loss')
    history = model.fit(
        train_X, train_Y,
        validation_data=(test_X, test_Y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[es, lr]
    )
    return history