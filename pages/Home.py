import streamlit as st

st.title("🏠 Home Page")

st.markdown("""
## 📌 Project Overview

This system detects violent scenes from uploaded videos using Deep Learning.

### 🎯 Objective
- Detect violent scenes
- Classify video as violent or non-violent
- Show prediction confidence

### 🧠 Model Used
CNN + LSTM Architecture

### ⚙ How It Works
1. Extract frames from video
2. Resize frames
3. Pass sequence into trained model
4. Get prediction
""")