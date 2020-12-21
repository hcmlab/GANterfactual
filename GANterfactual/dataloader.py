from __future__ import print_function, division

import os

import keras
import numpy as np


class DataLoader():
    def __init__(self, dataset_name=None, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

        self.image_gen_config = {
            "horizontal_flip": False,
            "preprocessing_function": (lambda x: x / 127.5 - 1.),
            "rescale": None,
        }

    def load_batch(self, train_N="NEGATIVE", train_P="POSITIVE", batch_size=16, is_testing=False):
        generator = keras.preprocessing.image.ImageDataGenerator(**self.image_gen_config)

        flow_args = dict(
            class_mode="categorical",
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=True,
            target_size=self.img_res,
        )

        subdir = "validation" if is_testing else "train"

        negative_path = os.path.join(self.dataset_name, subdir, train_N)
        positive_path = os.path.join(self.dataset_name, subdir, train_P)

        negative_flow = generator.flow_from_directory(negative_path, **flow_args)
        positive_flow = generator.flow_from_directory(positive_path, **flow_args)

        # endless loop so we can use the maximum
        n_batches = max(len(negative_flow), len(positive_flow))

        for b_normal, b_pneumo, _ in zip(negative_flow, positive_flow, range(n_batches)):
            normal, _ = b_normal
            pneumo, _ = b_pneumo

            yield normal, pneumo

    def load_single(self, path):
        img = keras.preprocessing.image.load_img(path, color_mode="grayscale", target_size=self.img_res)
        x = keras.preprocessing.image.img_to_array(img) / 127.5 - 1
        return x

    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        keras.preprocessing.image.save_img(path, x)
