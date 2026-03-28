import json

with open("train.ipynb", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("validation_steps=validation_steps", "validation_steps=None")

with open("train.ipynb", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated train.ipynb")
