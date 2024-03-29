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


def crop_and_resize(img, RESIZE, central_crop_frac = 0.9):
    
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
    #img = tf.reshape(img, [*TFREC_SIZES, 3])
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


def augment_images(image, label, resize_factor, crop_ratio = 0.9):
    
    #max_angle=tf.constant(np.pi/6)
    img = tf.image.random_flip_left_right(image)
    #img = tfa.image.rotate(img, angles=max_angle*tf.random.uniform([1], minval=-1, maxval=1, dtype=tf.dtypes.float32)) # added random rotation, 30 degrees each side
    img = tf.image.random_flip_up_down(image)
    img = tf.image.central_crop(image, central_fraction = crop_ratio)
    img = tf.image.resize( img, size = resize_factor)
    return img, label


def get_train_ds_tfrec(LS_FILENAMES, 
                       TFREC_DICT, 
                       TFREC_SIZES, 
                       RESIZE_FACTOR, 
                       NUM_CLASSES, 
                       BATCH_SIZE, 
                       DataRepeat = False, 
                       AugmentLayer = False, 
                       Nsuffle = 2048):

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

def get_train_ds_tfrec_from_dict(config_dict, 
                                 label_name, 
                                 image_key = "image", 
                                 imagenet_normalize = True, 
                                 DataRepeat = False, 
                                 AugmentLayer = False, 
                                 Nsuffle = 2048):
    
    LS_FILENAMES =  config_dict["ls_train_files"]
    TFREC_DICT =  config_dict["tfrec_structure"]
    TFREC_SIZES =  config_dict["tfrec_shape"]
    RESIZE_FACTOR =  config_dict["resize_resol"]
    NUM_CLASSES =  config_dict["N_cls"]
    BATCH_SIZE =  config_dict["batch_size"]
    
    if config_dict["crop_ratio"]:
        CROP_RATIO = config_dict["crop_ratio"]
    else:
        CROP_RATIO = 0.9
        
    
    tfrec_format = tfrec_format_generator(TFREC_DICT)
    dataset = load_tfrec_dataset(LS_FILENAMES, 
                                 tfrec_format = tfrec_format, 
                                 tfrec_sizes = TFREC_SIZES,
                                 label_name = label_name,
                                 image_key = image_key)
    
    dataset = dataset.map(lambda image, label: onehot(image, label, n_cls = NUM_CLASSES), num_parallel_calls=AUTO)
    if AugmentLayer:
        dataset = dataset.map(lambda image, label: (AugmentLayer(image), label), num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: augment_images(image, label, 
                                                              resize_factor = RESIZE_FACTOR, 
                                                              crop_ratio = CROP_RATIO), num_parallel_calls=AUTO).prefetch(AUTO)
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

def get_ds_tfrec_from_dict(tfrec_PATH, TFREC_DICT):

    tfrec_format = tfrec_format_generator(TFREC_DICT)
    raw_dataset = tf.data.TFRecordDataset(tfrec_PATH)
    
    def parse_tfrecord_fn(example, TFREC_FORMAT):    
        example = tf.io.parse_single_example(example, TFREC_FORMAT)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        return example
    parsed_dataset = raw_dataset.map(lambda x : parse_tfrecord_fn(x, tfrec_format))
    
    return parsed_dataset

### Affine transform, similarity trwansform, rigid transform
### Modified the code from kaggle notebookL source
import tensorflow.keras.backend as K

def get_mat(rotation, shear, zoom_hw, shift_hw ):
    import math
    # returns 3x3 transformmatrix which transforms indicies
    height_zoom, width_zoom = zoom_hw
    height_shift, width_shift =shift_hw
    height_shift = tf.constant([height_shift],dtype='float32')
    width_shift = tf.constant([width_shift],dtype='float32')

    # CONVERT DEGREES TO RADIANS
    rotation = tf.reshape(float(math.pi * rotation / 180.),[1])
    shear = tf.reshape(float(math.pi * shear / 180.),[1])
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    affine_matrix = tf.reshape( 
                      tf.concat([one/height_zoom, zero, height_shift, 
                                zero, one/width_zoom, width_shift, 
                                zero, zero, one],
                                         axis=0),
                              [3,3] )    
    
    return K.dot(rotation_matrix, affine_matrix)

# how to do reflection??

def transform(image, rotate = 0, shear = 0, zoom = 1, shift_hw = [0,0]):
    import math
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = image.shape[0]
    print(DIM)
    XDIM = DIM%2 #fix for size 331
    
    rot = rotate 
    shr = shear
    h_shift, w_shift = shift_hw
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,[zoom,zoom],[h_shift,w_shift]) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])

### Change this to image processing Layer! 
### Q. Batch time, how many random initializations are done?
class image_transform():
  import numpy as np
  def __init__(self,
               rotate = 0,
               shift_hw = [0,0],
                zoom = 1,
               shear = 0):
    self.rotate  = rotate
    self.shift_hw = shift_hw
    self.zoom = zoom
    self.shear = shear

  def __call__(self, image):
    im = transform(image,
              rotate = self.rotate *np.random.normal(size = 1),
              shift_hw = self.shift_hw* np.random.normal(size = 2),
              zoom = np.random.uniform(1, self.zoom, size=1), 
              shear = self.shear)
    return im
    
    
    
    


