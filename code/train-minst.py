import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from nn import relu, softmax, cross_entropy, init_weights, save_weights

def iterate_batches(X, Y, batch_size, rng):
    n = X.shape[0]
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], Y[idx]

#import the data 
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

#scale X 
X = X / 255.0

#shuffle the data so the unvanted sequence patterns are removed 
rng = np.random.default_rng(0)
shuffle = rng.permutation(len(X))
X, y = X[shuffle], y[shuffle]

#test train split 
n_test = 10000
X_train, X_test = X[:-n_test], X[-n_test:]
y_train, y_test = y[:-n_test], y[-n_test:]

#Converting the labeled Y to One hot encoded matrix  
Y_train = np.zeros((len(y_train), 10))
Y_train[np.arange(len(y_train)), y_train] = 1

print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("Y_train:", Y_train.shape, " y_test:", y_test.shape)

#set up the network
n_in, n_hidden, n_out = 784, 128, 10
W1 = init_weights(n_in, n_hidden, rng)
b1 = np.zeros(n_hidden)
W2 = init_weights(n_hidden, n_out, rng)
b2 = np.zeros(n_out)

lr = 0.5
batch_size = 128
epochs = 8

#train, with a timer so we can see how long each epoch (and the whole run) takes
#also keep a record of loss/accuracy per epoch so we can plot it afterwards
history = {"train_loss": [], "train_acc": [], "test_acc": []}
training_start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    for Xb, Yb in iterate_batches(X_train, Y_train, batch_size, rng):
        Z1 = Xb @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        P  = softmax(Z2)

        n = Xb.shape[0]
        dZ2 = (P - Yb) / n
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = Xb.T @ dZ1
        db1 = dZ1.sum(axis=0)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    #check progress once per epoch, on the full training set
    A1_full = relu(X_train @ W1 + b1)
    P_full  = softmax(A1_full @ W2 + b2)
    loss = cross_entropy(P_full, Y_train)
    acc  = (P_full.argmax(axis=1) == y_train).mean()

    #also check test accuracy every epoch (not just at the very end) so we can
    #see on the plot if the network starts overfitting partway through training
    A1_test_epoch = relu(X_test @ W1 + b1)
    P_test_epoch  = softmax(A1_test_epoch @ W2 + b2)
    test_acc_epoch = (P_test_epoch.argmax(axis=1) == y_test).mean()

    history["train_loss"].append(loss)
    history["train_acc"].append(acc)
    history["test_acc"].append(test_acc_epoch)

    epoch_time = time.time() - epoch_start
    print(f"epoch {epoch+1}/{epochs}  loss {loss:.4f}  train acc {acc:.3f}  test acc {test_acc_epoch:.3f}  ({epoch_time:.1f}s)")

total_time = time.time() - training_start
print(f"\ntraining finished in {total_time:.1f}s total ({total_time/epochs:.1f}s/epoch average)")

#the moment of truth: accuracy on data the network never saw
A1_test = relu(X_test @ W1 + b1)
P_test  = softmax(A1_test @ W2 + b2)
test_acc = (P_test.argmax(axis=1) == y_test).mean()
print(f"final test accuracy: {test_acc:.3f}")

#plot the training curves so we can see it learning instead of just reading numbers
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(history["train_loss"])
axes[0].set_title("Training Loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["test_acc"], label="test")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")
axes[1].legend()

plt.tight_layout()
out_path = Path(__file__).parent / "training_curves.png"
plt.savefig(out_path)
print("saved training curves to", out_path)

#save the trained weights so the Streamlit app can load them without retraining
weights_path = Path(__file__).parent.parent / "model" / "weights.npz"
weights_path.parent.mkdir(exist_ok=True)
save_weights(weights_path, W1, b1, W2, b2)
print("saved trained weights to", weights_path)