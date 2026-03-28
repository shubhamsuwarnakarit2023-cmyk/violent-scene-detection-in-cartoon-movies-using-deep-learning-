import os
import tensorflow
from keras.layers import InputLayer, TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import load_model, model_from_json, Sequential
from tensorflow.keras.optimizers import Adam
import time
from datetime import datetime
import keras
from TimeDistributedImageDataGenerator import TimeDistributedImageDataGenerator
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

batch_size = 16
frame_sequence = 16

model = Sequential()
model.add(TimeDistributed(Conv2D(64,(3,3),padding='same', strides=(2,2), activation='relu'), input_shape=(frame_sequence, 224, 224, 3)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', strides=(2,2), activation='relu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', strides=(2,2), activation='relu')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(5, activation='softmax'))

model.summary()

datagen = TimeDistributedImageDataGenerator(time_steps=16)

train_it = datagen.flow_from_directory(
    'Dataset/train_set/',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)

val_it = datagen.flow_from_directory(
    'Dataset/val_set/',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)

batchX, batchy = train_it[0]
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

opt = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

filepath = 'saved_models/model-ep{epoch:02d}-loss{loss:.3f}-val_loss{val_loss:.3f}.keras'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

if os.path.exists('best_model1.h5'):
    try:
        model.load_weights('best_model1.h5')
        print("Loaded existing weights from best_model1.h5")
    except Exception as e:
        print("Could not load weights:", e)

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

print("Starting training loop...")
start_time = time.perf_counter()

# Train for 1 epoch to demonstrate functionality without taking hours
history = model.fit(
        train_it,
        steps_per_epoch=5,
        initial_epoch=0,
        epochs=1,
        validation_data=val_it,
        validation_steps=2,
        callbacks=[tensorboard_callback, checkpoint],
        verbose=1)

print("Training finished. Time taken:", time.perf_counter() - start_time)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("best_model1.weights.h5")
print("Saved best_model1.weights.h5 to disk")
