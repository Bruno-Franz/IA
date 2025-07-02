"""Funções de redes neurais reutilizáveis para dados tabulares, texto e imagem."""

from __future__ import annotations

import time
from typing import Tuple, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


# ---------------------- Tabular Data ----------------------

def treinar_mlp_sklearn(X_train: np.ndarray, X_test: np.ndarray,
                        y_train: Iterable, y_test: Iterable,
                        hidden_layer_sizes=(100,), max_iter=300) -> dict:
    # treino de MLP usando scikit-learn
    """Train a scikit-learn MLPClassifier and return metrics."""
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)

    return {
        "method": "MLPClassifier",
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "duration": time.time() - start,
    }


def treinar_mlp_keras(X_train: np.ndarray, X_test: np.ndarray,
                      y_train: Iterable, y_test: Iterable,
                      epochs: int = 50) -> dict:
    """Train a simple Keras MLP for binary classification."""
    # rede simples em Keras
    start = time.time()
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=5,
                                                        restore_best_weights=True)],
              verbose=0)

    loss, _ = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)
    return {
        "method": "Keras MLP",
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "loss": loss,
        "duration": time.time() - start,
    }


# ---------------------- Image Data ----------------------

IMG_SIZE = (180, 180)
BATCH_SIZE = 32

def _preparar_ds_imagem(ds):
    # redimensiona e normaliza imagens
    def resize_norm(img, label):
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label
    return ds.map(resize_norm).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def treinar_cnn_simples(ds_train, ds_val, ds_test, num_classes: int,
                        epochs: int = 10) -> dict:
    """Train a small CNN for image classification."""
    # CNN simples para imagens
    start = time.time()
    model = keras.Sequential(
        [
            layers.Conv2D(32, 3, activation="relu", input_shape=(*IMG_SIZE, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(_preparar_ds_imagem(ds_train), validation_data=_preparar_ds_imagem(ds_val),
              epochs=epochs, verbose=0)

    loss, _ = model.evaluate(_preparar_ds_imagem(ds_test), verbose=0)
    y_true = []
    y_pred = []
    for batch, labels in ds_test:
        preds = model.predict(batch, verbose=0)
        y_pred.extend(preds.argmax(axis=1))
        y_true.extend(labels.numpy())
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "method": "Simple CNN",
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "loss": loss,
        "duration": time.time() - start,
    }


def treinar_cnn_profundo(ds_train, ds_val, ds_test, num_classes: int,
                         epochs: int = 20) -> dict:
    """Train a deeper CNN with additional layers."""
    # versão mais profunda da CNN
    start = time.time()
    model = keras.Sequential(
        [
            layers.Conv2D(32, 3, activation="relu", input_shape=(*IMG_SIZE, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(
        _preparar_ds_imagem(ds_train),
        validation_data=_preparar_ds_imagem(ds_val),
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                 restore_best_weights=True)],
        verbose=0,
    )

    loss, _ = model.evaluate(_preparar_ds_imagem(ds_test), verbose=0)
    y_true = []
    y_pred = []
    for batch, labels in ds_test:
        preds = model.predict(batch, verbose=0)
        y_pred.extend(preds.argmax(axis=1))
        y_true.extend(labels.numpy())
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "method": "Deep CNN",
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "loss": loss,
        "duration": time.time() - start,
    }


# ---------------------- Text Data ----------------------

def treinar_cnn_texto(train_seq: np.ndarray, test_seq: np.ndarray,
                      y_train: Iterable, y_test: Iterable,
                      vocab_size: int, embedding_dim: int = 16,
                      epochs: int = 10) -> dict:
    """CNN model for text classification."""
    # CNN para classificação de texto
    start = time.time()
    max_length = train_seq.shape[1]
    model = keras.Sequential(
        [
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ]
    )

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_seq, y_train, epochs=epochs, validation_split=0.2, verbose=0)

    loss, _ = model.evaluate(test_seq, y_test, verbose=0)
    probs = model.predict(test_seq, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)
    return {
        "method": "CNN Text",
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "loss": loss,
        "duration": time.time() - start,
    }


def treinar_cnn_lstm_texto(train_seq: np.ndarray, test_seq: np.ndarray,
                            y_train: Iterable, y_test: Iterable,
                            vocab_size: int, embedding_dim: int = 32,
                            epochs: int = 10) -> dict:
    """CNN + LSTM model for text classification."""
    # combinação de CNN e LSTM para texto
    start = time.time()
    max_length = train_seq.shape[1]
    model = keras.Sequential(
        [
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(),
            layers.LSTM(64),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ]
    )

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_seq, y_train, epochs=epochs, validation_split=0.2,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                       restore_best_weights=True)],
              verbose=0)

    loss, _ = model.evaluate(test_seq, y_test, verbose=0)
    probs = model.predict(test_seq, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)
    return {
        "method": "CNN+LSTM Text",
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "loss": loss,
        "duration": time.time() - start,
    }


def treinar_gru_texto(train_seq: np.ndarray, test_seq: np.ndarray,
                      y_train: Iterable, y_test: Iterable,
                      vocab_size: int, embedding_dim: int = 32,
                      epochs: int = 10) -> dict:
    """GRU based model for text classification."""
    # modelo baseado em GRU
    start = time.time()
    max_length = train_seq.shape[1]
    model = keras.Sequential(
        [
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            layers.GRU(64),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ]
    )

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_seq, y_train, epochs=epochs, validation_split=0.2,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                       restore_best_weights=True)],
              verbose=0)

    loss, _ = model.evaluate(test_seq, y_test, verbose=0)
    probs = model.predict(test_seq, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)
    return {
        "method": "GRU Text",
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "loss": loss,
        "duration": time.time() - start,
    }

