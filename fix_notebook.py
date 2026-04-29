import json

file_path = "c:\\Detection-of-Violent-Scenes-in-Cartoon-Movies-main\\train.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "loss, acc = model.evaluate(val_gen, steps=validation_steps, verbose=1)" in source and "model.compile" not in source:
            # We want to insert model.compile before model.evaluate
            new_source = []
            for line in cell['source']:
                if "loss, acc = model.evaluate" in line:
                    new_source.append("model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\\n\\n")
                new_source.append(line)
            cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
