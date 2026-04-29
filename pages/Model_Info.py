import streamlit as st

st.title("🧠 Model Information")

st.markdown("""
## Architecture

- CNN for spatial feature extraction
- LSTM for temporal sequence learning

## Training Details

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 2 (can increase to improve accuracy)
- Frame Sequence Length: 8

## Current Status

Model accuracy needs improvement.
Future improvement: more dataset + more epochs.
""")