__copyright__ = """
Copyright (c) 2022 John Park, Dimitre Oliveira
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
### Adapted part from Dimitre Oliveria's code below:
### https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/creating_tfrecords.py

import numpy as np
import pandas as pd
import re
import tensorflow as tf
import tensorflow_addons as tfa
import json


AUTO = tf.data.experimental.AUTOTUNE

def json_open(full_path):
    with open(full_path) as f:
        out = json.load(f)
    return out

def json_save(jsonData, filename):
    with open(filename, 'w') as outfile:
        json.dump(jsonData, outfile, sort_keys = True, indent = 4,
               ensure_ascii = False)


def load_image_and_resize(full_path, RESIZE, central_crop_frac = 0.9):
    
    raw = tf.io.read_file(full_path)
    img = tf.io.decode_image(raw)
    img = tf.image.central_crop(img, central_fraction = central_crop_frac)
    img = tf.image.resize(img, size = RESIZE)
    return(img)


def tfrec_format_generator(dictionary_obj):

    tfrec_format= dict()
    for key, value in dictionary_obj.items():
        if value == "str":
            tfrec_format[key] = tf.io.FixedLenFeature([], tf.string)
        elif value == "int":
            tfrec_format[key] = tf.io.FixedLenFeature([], tf.int64)
        elif value == "float": #Not tested
            tfrec_format[key] = tf.io.FixedLenFeature([], tf.int64)
    return tfrec_format


def parse_tfrecord_fn(example, TFREC_FORMAT):    
    
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    #tf.sparse.to_dense(example["bbox"])
    return example


def decode_image(image_data_bytes, TFREC_SIZES):

    img = tf.io.decode_jpeg(image_data_bytes, channels=3)  # image format uint8 [0,255]
    img = tf.reshape(img, [*TFREC_SIZES, 3])
    img = tf.cast(img, dtype = tf.float32) / 255.0
    return img


def read_tfrecord(example, TFREC_FORMAT, TFREC_SIZES, LABEL_NAME, IMAGE_NAME="image"):

    LABELED_TFREC_FORMAT = TFREC_FORMAT
    #{ EXAMPLE OF TFREC FORMAT 
    #    "image": tf.io.FixedLenFeature([], tf.string),
    #    "image_id": tf.io.FixedLenFeature([], tf.string),
    #    "spname": tf.io.FixedLenFeature([], tf.int64),
    #    "genus": tf.io.FixedLenFeature([], tf.int64),
    #    "family": tf.io.FixedLenFeature([], tf.int64)
    #}
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example[IMAGE_NAME], TFREC_SIZES = TFREC_SIZES)
    label = example[LABEL_NAME]
    #hierarchy = how can you get the hierarchy? 
    return image, label


def load_tfrec_dataset(filenames, tfrec_sizes, tfrec_format, label_name, image_key,  labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(lambda Example: read_tfrecord(Example, 
                                                        TFREC_SIZES = tfrec_sizes,
                                                        TFREC_FORMAT = tfrec_format,
                                                        LABEL_NAME = label_name,
                                                        IMAGE_NAME = image_key))# if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset


def onehot(image, label, n_cls): #one hot encoding of labels. n_cls denotes number of classes. 
    
    return image,tf.one_hot(label, n_cls)


def normalize_to_ImageNet(image): # Preprocess with Imagenet's mean and stddev:
   
    image -= tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=image.dtype)
    return image


def normalize_RGB(image, label):
    
    image = normalize_to_ImageNet(image)
    return image, label


def augment_images(image, label, resize_factor):
    
    max_angle=tf.constant(np.pi/6)
    img = tf.image.random_flip_left_right(image)
    img = tfa.image.rotate(img, angles=max_angle*tf.random.uniform([1], minval=-1, maxval=1, dtype=tf.dtypes.float32)) # added random rotation, 30 degrees each side
    img = tf.image.central_crop(image, central_fraction = 0.9)
    img = tf.image.resize( img, size = resize_factor)
    return img, label


def get_train_ds_tfrec(LS_FILENAMES, TFREC_DICT, TFREC_SIZES, RESIZE_FACTOR, NUM_CLASSES, BATCH_SIZE, DataRepeat = False, AugmentLayer = False, Nsuffle = 2048):

    tfrec_format = tfrec_format_generator(TFREC_DICT)
    dataset = load_tfrec_dataset(LS_FILENAMES, tfrec_format = tfrec_format, tfrec_sizes = TFREC_SIZES)
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    if AugmentLayer:
        dataset = dataset.map(lambda image, label: (AugmentLayer(image), label), num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    #dataset = dataset.repeat() # the training dataset must repeat for several epochs #Check what default value is! 
    if DataRepeat == True:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(Nsuffle)
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_train_ds_tfrec_from_dict(config_dict, label_name, image_key = "image", imagenet_normalize = True, DataRepeat = False, AugmentLayer = False, Nsuffle = 2048):
    
    LS_FILENAMES =  config_dict["ls_train_files"]
    TFREC_DICT =  config_dict["tfrec_structure"]
    TFREC_SIZES =  config_dict["tfrec_shape"]
    RESIZE_FACTOR =  config_dict["resize_resol"]
    NUM_CLASSES =  config_dict["N_cls"]
    BATCH_SIZE =  config_dict["batch_size"]
    
    tfrec_format = tfrec_format_generator(TFREC_DICT)
    dataset = load_tfrec_dataset(LS_FILENAMES, 
                                 tfrec_format = tfrec_format, 
                                 tfrec_sizes = TFREC_SIZES,
                                 label_name = label_name,
                                 image_key = image_key)
    
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    if AugmentLayer:
        dataset = dataset.map(lambda image, label: (AugmentLayer(image), label), num_parallel_calls=AUTO).prefetch(AUTO)
    if imagenet_normalize:
        dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    if DataRepeat == True:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(Nsuffle)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # set drop remainder true for kecam test
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_vali_ds_tfrec(LS_FILENAMES, TFREC_DICT, TFREC_SIZES, RESIZE_FACTOR, NUM_CLASSES, BATCH_SIZE, imagenet_normalize = True, AugmentLayer = False):

    tfrec_format = tfrec_format_generator(TFREC_DICT)
    dataset = load_tfrec_dataset(LS_FILENAMES, tfrec_format = tfrec_format, tfrec_sizes = TFREC_SIZES)
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    if imagenet_normalize:
        dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_vali_ds_tfrec_from_dict(config_dict, label_name,  image_key = "image", AugmentLayer = False):
    
    LS_FILENAMES =  config_dict["ls_vali_files"]
    TFREC_DICT =  config_dict["tfrec_structure"]
    TFREC_SIZES =  config_dict["tfrec_shape"]
    RESIZE_FACTOR =  config_dict["resize_resol"]
    NUM_CLASSES =  config_dict["N_cls"]
    BATCH_SIZE =  config_dict["batch_size"]
    
    tfrec_format = tfrec_format_generator(TFREC_DICT)
    dataset = load_tfrec_dataset(LS_FILENAMES, 
                                 tfrec_format = tfrec_format, 
                                 tfrec_sizes = TFREC_SIZES,
                                 label_name = label_name,
                                 image_key = image_key)
    
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def count_dps_in_tfrec(filenames):
    # count data points in tfrec
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def get_filenames(DS_PATH):
    TRAINING_FILENAMES = tf.io.gfile.glob(DS_PATH + '/*train*.tfrec')#+tf.io.gfile.glob(GCS_DS_PATH_2 + '/*.tfrec')+tf.io.gfile.glob(GCS_DS_PATH_3 + '/*.tfrec')
    VALIDATION_FILENAMES = tf.io.gfile.glob(DS_PATH + '/*val*.tfrec')
    NUM_TRAINING_IMAGES = count_dps_in_tfrec(TRAINING_FILENAMES)
    NUM_VALIDATION_IMAGES = count_dps_in_tfrec(VALIDATION_FILENAMES)
    print('Dataset: {} training images, {} validation images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))
    return TRAINING_FILENAMES, VALIDATION_FILENAMES, NUM_TRAINING_IMAGES
    


