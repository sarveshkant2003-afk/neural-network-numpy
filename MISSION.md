# Mission: Neural Networks from Scratch with NumPy

## Why
Sarvesh wants deep conceptual mastery of how neural networks actually work — not just calling `model.fit()` in PyTorch/TensorFlow. Building every piece (forward pass, backprop, gradient descent) by hand in raw numpy is the forcing function: if you can't implement it, you don't really understand it.

## Success looks like
- Can explain and derive the forward pass and backpropagation for a feedforward network by hand (chain rule, matrix shapes) before writing the code for it.
- Has a working feedforward neural network, implemented only with numpy (no autograd/ML frameworks), trained on the MNIST digit dataset in VS Code.
- Can explain every line of their own implementation: layer initialization, activation functions, loss function, gradient computation, weight updates.
- Comfortable with the numpy operations (broadcasting, `@`/`dot`, vectorization, array shapes) needed to implement it, since this was a gap going in.

## Constraints
- No deadline — steady pace, a handful of focused sessions over the coming weeks.
- Working in VS Code on their own machine; lessons should produce/point to code files runnable there, not just browser exercises.

## Out of scope (for now)
- Convolutional/recurrent/transformer architectures — feedforward first.
- ML frameworks (PyTorch, TensorFlow, JAX) — the whole point is doing it without them.
- Deployment, production serving, or MLOps concerns.
