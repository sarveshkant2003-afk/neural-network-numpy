import numpy as np 


def make_toy_dataset(n_per_class=50, n_features=4, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, size=(n_classes, n_features))
    X, labels = [], []
    for c in range(n_classes):
        pts = centers[c] + rng.normal(scale=0.5, size=(n_per_class, n_features))
        X.append(pts)
        labels += [c] * n_per_class
    return np.vstack(X), np.array(labels)


def relu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    shifted = Z - Z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(P, Y, eps=1e-12):
    P = np.clip(P, eps, 1 - eps)
    return np.mean(-np.sum(Y * np.log(P), axis=1))

np.random.seed(0)
n_in, n_hidden, n_out = 4, 8, 3
X, labels = make_toy_dataset(n_features=n_in, n_classes=n_out)

Y = np.zeros((len(labels), n_out))
Y[np.arange(len(labels)), labels] = 1

W1 = np.random.randn(n_in, n_hidden) * 0.1
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, n_out) * 0.1
b2 = np.zeros(n_out)

lr = 0.1
batch = X.shape[0]

for step in range(201):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    P  = softmax(Z2)
    loss = cross_entropy(P, Y)

    dZ2 = (P - Y) / batch
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if step % 20 == 0:
        acc = (P.argmax(axis=1) == labels).mean()
        print(f"step {step:4d}  loss {loss:.4f}  accuracy {acc:.2f}")