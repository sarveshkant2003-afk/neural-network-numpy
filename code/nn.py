"""Shared numpy network code: the same math from Lessons 2-6, used by both
the training script and the Streamlit app so the forward pass isn't duplicated."""
import numpy as np


def relu(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    shifted = Z - Z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(P, Y, eps=1e-12):
    P = np.clip(P, eps, 1 - eps)
    return np.mean(-np.sum(Y * np.log(P), axis=1))


def init_weights(n_in, n_out, rng):
    return rng.standard_normal((n_in, n_out)) * np.sqrt(1.0 / n_in)


def forward(X, W1, b1, W2, b2):
    """Returns (A1, P): hidden activations and output probabilities."""
    A1 = relu(X @ W1 + b1)
    P = softmax(A1 @ W2 + b2)
    return A1, P


def save_weights(path, W1, b1, W2, b2):
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)


def load_weights(path):
    data = np.load(path)
    return data["W1"], data["b1"], data["W2"], data["b2"]
