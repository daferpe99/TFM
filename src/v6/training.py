import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import build_model_tuner

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

def train_model(model, x_text_train, x_text_test, x_num_train, x_num_test, y_train, y_test, class_weight):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    
    """Entrena el modelo y retorna el historial."""
    history = model.fit(
    {"text_input": x_text_train, "numeric_input": x_num_train},
    y_train,
    validation_data=(
        {"text_input": x_text_test, "numeric_input": x_num_test},
        y_test
    ),
    epochs=10,
    batch_size=32,
    callbacks=[callbacks],  # tus callbacks actuales
    class_weight=class_weight)

    return history

def cross_validate_model(build_model_fn, x_text, x_numeric, y, n_splits=5, batch_size=32, epochs=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(x_text, y), 1):
        print(f"\n🔁 Fold {fold}")
        x_text_train, x_text_test = x_text[train_idx], x_text[test_idx]
        x_num_train, x_num_test = x_numeric[train_idx], x_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, text_vectorizer = build_model_fn(n_numeric_features=x_numeric.shape[1])
        text_vectorizer.adapt(x_text_train)

        model.fit(
            {"text_input": x_text_train, "numeric_input": x_num_train},
            y_train,
            validation_data=({"text_input": x_text_test, "numeric_input": x_num_test}, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        y_pred = (model.predict({"text_input": x_text_test, "numeric_input": x_num_test}) > 0.5).astype(int)
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))

    # Mostrar resultados promediados
    for metric, values in metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return metrics

def cross_validate_with_best_hp(best_hp, x_text, x_numeric, y, n_splits, batch_size, epochs, text_vectorizer):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(x_text, y), 1):
        print(f"\n🔁 Fold {fold}")

        x_text_train, x_text_test = x_text[train_idx], x_text[test_idx]
        x_num_train, x_num_test = x_numeric[train_idx], x_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, _ = build_model_tuner(best_hp, n_numeric_features=x_numeric.shape[1], text_vectorizer=text_vectorizer)

        model.fit(
            {"text_input": x_text_train, "numeric_input": x_num_train},
            y_train,
            validation_data=(
                {"text_input": x_text_test, "numeric_input": x_num_test}, y_test
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            verbose=0
        )

        y_pred = (model.predict({"text_input": x_text_test, "numeric_input": x_num_test}) > 0.5).astype(int)
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))

    # Mostrar resumen
    for metric, values in metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return metrics