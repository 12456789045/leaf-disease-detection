from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.utils import load_img, img_to_array, save_img

datagen = ImageDataGenerator(rotation_range=25, zoom_range=0.2, horizontal_flip=True)

for folder in os.listdir("dataset"):
    path = f"dataset/{folder}"

    for img in os.listdir(path):
        img_path = f"{path}/{img}"

        image = load_img(img_path, target_size=(224, 224))
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            save_img(f"{path}/aug_{i}_{img}", batch[0])
            i += 1
            if i > 3:
                break
