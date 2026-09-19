"""Streamlit demo: draw a digit, get a live prediction from a neural network
built entirely with numpy (no ML frameworks) -- trained in code/train-minst.py."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

sys.path.append(str(Path(__file__).parent / "code"))
from nn import forward, load_weights

WEIGHTS_PATH = Path(__file__).parent / "model" / "weights.npz"

st.set_page_config(page_title="Numpy Neural Network - MNIST", page_icon="✍️")


@st.cache_resource
def get_weights():
    return load_weights(WEIGHTS_PATH)


def canvas_to_input(image_data):
    """RGBA canvas array -> normalized, flattened (1, 784) input, MNIST-style."""
    img = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    x = np.array(img, dtype=np.float64) / 255.0
    return x.reshape(1, 784), x


st.title("Draw a Digit")
st.caption(
    "A feedforward neural network (784 → 128 → 10) implemented from scratch "
    "with numpy — no PyTorch, no TensorFlow. Draw a digit below and it predicts what it is."
)

if not WEIGHTS_PATH.exists():
    st.error(
        f"No trained weights found at `{WEIGHTS_PATH.relative_to(Path(__file__).parent)}`. "
        "Run `python code/train-minst.py` first to train the network and save its weights."
    )
    st.stop()

W1, b1, W2, b2 = get_weights()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw here")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    if canvas_result.image_data is not None and canvas_result.image_data[:, :, 3].max() > 0:
        X, preview = canvas_to_input(canvas_result.image_data)
        _, P = forward(X, W1, b1, W2, b2)
        pred = int(P.argmax())

        st.markdown(f"## Predicted: **{pred}**")
        st.caption(f"Confidence: {P[0, pred] * 100:.1f}%")

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.bar(range(10), P[0])
        ax.set_xticks(range(10))
        ax.set_xlabel("digit")
        ax.set_ylabel("probability")
        st.pyplot(fig)

        with st.expander("What the network actually sees (28×28, downsampled)"):
            st.image(preview, width=140, clamp=True)
    else:
        st.info("Draw a digit on the left to see a prediction.")

st.divider()
st.subheader("What the first layer learned")
st.caption(
    "Each of the 128 hidden units has its own 784-number weight vector — one weight per "
    "input pixel. Reshaped back into 28×28, a few of them look like blurry pen-strokes: "
    "the patterns this layer learned to detect."
)

n_show = 16
fig, axes = plt.subplots(2, 8, figsize=(10, 2.6))
for i, ax in enumerate(axes.flat):
    ax.imshow(W1[:, i].reshape(28, 28), cmap="gray")
    ax.axis("off")
st.pyplot(fig)

st.caption(
    "Built while learning neural networks from scratch — "
    "see the full lesson series in `lessons/` and `reference/` in this repo."
)
