# Neural Network from Scratch (NumPy)

A feedforward neural network (784 → 128 → 10) implemented entirely with NumPy — no PyTorch, no TensorFlow. Trained on MNIST to classify handwritten digits, with a Streamlit app to draw a digit and get a live prediction.

I implemented the neural network from scratch using only NumPy to understand what actually happens behind the scenes rather than relying on high-level libraries. This approach helped me gain a clear understanding of the core concepts such as forward propagation, activation functions, loss calculation, backpropagation, gradients, and weight updates. By building each component myself, I was able to see how the mathematical concepts translate into code and develop a stronger foundation for understanding more advanced deep learning frameworks.

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
