import streamlit as st

st.title("📂 Dataset Information")

st.markdown("""
## Dataset Structure

Train Classes:
- bloody
- explosions
- fight
- gun shot
- non-violence
- violence

Validation Classes:
- non-violence
- violence

## Future Improvements

- Collect more clips
- Balance dataset
- Train for 20+ epochs
- Use transfer learning
""")