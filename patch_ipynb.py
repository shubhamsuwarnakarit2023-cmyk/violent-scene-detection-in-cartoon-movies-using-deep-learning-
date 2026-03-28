import json

with open("train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        # Check if this is the model architecture cell
        is_model_cell = any("model = Sequential()" in "".join(cell["source"]) for _ in [1])
        if is_model_cell:
            new_source = '''from keras.layers import InputLayer, TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D
model = Sequential()
#input layer
model.add(
    TimeDistributed(
        Conv2D(64,(3,3),padding='same', strides=(2,2), activation='relu'),
      input_shape=(frame_sequence, 224, 224, 3)
    )
)
model.add(TimeDistributed(BatchNormalization()))

# first conv, 64
model.add(
    TimeDistributed( 
        Conv2D(64, (3,3), 
            padding='same', strides=(2,2), activation='relu')
    )
)
model.add(TimeDistributed(BatchNormalization()))

#pooling
model.add(
    TimeDistributed(
        MaxPooling2D((2,2), strides=(2,2))
    )
)

# Second conv, 128
model.add(
    TimeDistributed(
        Conv2D(128, (3,3),
            padding='same', strides=(2,2), activation='relu')
    )
)
model.add(TimeDistributed(BatchNormalization()))
 
#pooling
model.add(
    TimeDistributed(
        MaxPooling2D((2,2), strides=(2,2))
    )
)

model.add(TimeDistributed(Flatten()))

model.add(
    LSTM(512, return_sequences=False)
)
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(5, activation='softmax'))

model.summary()'''
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
            continue
            
        # For other cells, just iterate and replace strings
        for i, line in enumerate(cell["source"]):
            if "SGD" in line:
                cell["source"][i] = line.replace("SGD", "Adam")

with open("train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("train.ipynb successfully updated.")
