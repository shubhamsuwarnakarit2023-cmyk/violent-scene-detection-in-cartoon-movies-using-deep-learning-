from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import numpy as np

class TDGeneratorWrapper(Sequence):
    def __init__(self, generator, batch_size, time_steps):
        self.generator = generator
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.class_indices = generator.class_indices

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        x, y = self.generator[index]
        # Dynamically calculate actual batch size to prevent end-of-epoch modulo crashes
        actual_batch = x.shape[0] // self.time_steps
        
        # Drop any leftover frames that don't form a complete time_step sequence
        valid_frames = actual_batch * self.time_steps
        x = x[:valid_frames]
        y = y[:valid_frames]
        
        x = x.reshape((actual_batch, self.time_steps) + x.shape[1:])
        y = y.reshape((actual_batch, self.time_steps) + y.shape[1:])
        return x, y[:,0]

    def on_epoch_end(self):
        if hasattr(self.generator, 'on_epoch_end'):
            self.generator.on_epoch_end()

class TimeDistributedImageDataGenerator:


    def __init__(self, time_steps=8):
        self.time_steps = time_steps
        self.datagen = ImageDataGenerator(rescale=1./255)

    def flow_from_directory(self, directory, target_size=(224,224), batch_size=8, class_mode='categorical'):

        generator = self.datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size*self.time_steps,
            class_mode=class_mode
        )

        return TDGeneratorWrapper(generator, batch_size, self.time_steps)