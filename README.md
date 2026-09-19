# Neural Network from Scratch (NumPy)

A feedforward neural network (784 → 128 → 10) implemented entirely with NumPy — no PyTorch, no TensorFlow. Trained on MNIST to classify handwritten digits, with a Streamlit app to draw a digit and get a live prediction.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python code/train-minst.py
```

Downloads MNIST, trains the network, and saves the trained weights to `model/weights.npz` (needed by the app below).

## Run the app

```bash
streamlit run app.py
```

Draw a digit and see the network's prediction, plus a look at what its first layer learned.

## Project structure

- `code/nn.py` — the network itself: forward pass, activations, weight init/save/load
- `code/train-minst.py` — loads MNIST, trains, saves weights
- `app.py` — Streamlit demo

More to come.
