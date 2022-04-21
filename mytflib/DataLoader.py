### DataLoader for tensorflow ###
### Written and put together by John Park
### MIT license ###

import numpy as np
import pandas as pd
import re, math
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow as tf, re, math

AUTO = tf.data.experimental.AUTOTUNE


def tfrec_format_generator(dictionary_obj):

    tfrec_format= dict()
    for key, value in dictionary_obj.items():
        if value == "string":
            tfrec_format[key] = tf.io.FixedLenFeature([], tf.string)
        elif value == "int":
            tfrec_format[key] = tf.io.FixedLenFeature([], tf.int64)
        elif value == "float": #Not tested
            tfrec_format[key] = tf.io.FixedLenFeature([], tf.int64)
    return tfrec_format


def decode_image(image_data_bytes, tfrec_image_sizes):

    img = tf.image.decode_jpeg(image_data_bytes, channels=3)  # image format uint8 [0,255]
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [*tfrec_image_sizes, 3])
    return img


def read_tfrecord(example, TFREC_FORMAT, TFREC_SIZES , LABEL_NAME="spname", IMAGE_NAME="image"):

    LABELED_TFREC_FORMAT = TFREC_FORMAT
    #{ EXAMPLE OF TFREC FORMAT 
    #    "image": tf.io.FixedLenFeature([], tf.string),
    #    "image_id": tf.io.FixedLenFeature([], tf.string),
    #    "spname": tf.io.FixedLenFeature([], tf.int64),
    #    "genus": tf.io.FixedLenFeature([], tf.int64),
    #    "family": tf.io.FixedLenFeature([], tf.int64)
    #}
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example[IMAGE_NAME], tfrec_image_sizes = TFREC_SIZES )
    label = example[LABEL_NAME]
    #hierarchy = how can you get the hierarchy? 
    return image, label


def load_tfrec_dataset(filenames, tfrec_format, tfrec_sizes, label_name, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(lambda Example: read_tfrecord(Example, 
                                                        TFREC_FORMAT = tfrec_format, 
                                                        TFREC_SIZES = tfrec_sizes,
                                                        LABEL_NAME = label_name))# if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
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
    img = tfa.image.rotate(img,angles=max_angle*tf.random.uniform([1], minval=-1, maxval=1, dtype=tf.dtypes.float32)) # added random rotation, 30 degrees each side
    img = tf.image.central_crop(image, central_fraction = 0.9)
    img = tf.image.resize( img, size = resize_factor)
    return img, label


def get_train_ds_tfrec(LS_FILENAMES, TFREC_DICT, TFREC_SIZES, RESIZE_FACTOR, NUM_CLASSES, BATCH_SIZE, DataRepeat = False, MoreAugment = False, Nsuffle = 2048):

    tfrec_format = tfrec_format_generator(TFREC_DICT)
    dataset = load_tfrec_dataset(LS_FILENAMES, tfrec_format = tfrec_format, tfrec_sizes = TFREC_SIZES)
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    #dataset = dataset.repeat() # the training dataset must repeat for several epochs #Check what default value is! 
    if DataRepeat == True:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(Nsuffle)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_train_ds_tfrec_from_dict(config_dict, label_name, DataRepeat = False, MoreAugment = False, Nsuffle = 2048):
    
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
                                 label_name = label_name)
    
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    if DataRepeat == True:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(Nsuffle)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_vali_ds_tfrec(LS_FILENAMES, TFREC_DICT, TFREC_SIZES, RESIZE_FACTOR, NUM_CLASSES, BATCH_SIZE, MoreAugment = False):

    tfrec_format = tfrec_format_generator(TFREC_DICT)
    dataset = load_tfrec_dataset(LS_FILENAMES, tfrec_format = tfrec_format, tfrec_sizes = TFREC_SIZES)
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_vali_ds_tfrec_from_dict(config_dict, label_name, MoreAugment = False):
    
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
                                 label_name = label_name)
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    dataset = dataset.map(normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def count_tfrec_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def get_filenames(DS_PATH):
    TRAINING_FILENAMES = tf.io.gfile.glob(DS_PATH + '/*train*.tfrec')#+tf.io.gfile.glob(GCS_DS_PATH_2 + '/*.tfrec')+tf.io.gfile.glob(GCS_DS_PATH_3 + '/*.tfrec')
    VALIDATION_FILENAMES = tf.io.gfile.glob(DS_PATH + '/*val*.tfrec')
    NUM_TRAINING_IMAGES = count_tfrec_items(TRAINING_FILENAMES)
    NUM_VALIDATION_IMAGES = count_tfrec_items(VALIDATION_FILENAMES)
    print('Dataset: {} training images, {} validation images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))
    return TRAINING_FILENAMES, VALIDATION_FILENAMES, NUM_TRAINING_IMAGES
    

