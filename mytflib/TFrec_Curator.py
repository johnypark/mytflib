__copyright__ = """
Copyright (c) 2022 John Park
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

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os 
import glob
import random
import cv2
from mytflib.DataLoader import (
    tfrec_format_generator,
    parse_tfrecord_fn
)

# this file is for TFREC writer function.


def get_hchy_table(df, list_target_names):

    subset_class = dict()
    subset_hier = dict()
    
    for cls in list_target_names:
        subset_class[cls] = sorted(set(df[cls]))
        subset_hier[cls] = dict()
        idx = 0
        for ele in subset_class[cls]:
            subset_hier[cls][ele] = idx 
            idx += 1 
            
    table_hchy = subset_hier
    return(table_hchy)


## WRITE TFRECORD ##
 
## adapted from here: https://www.kaggle.com/code/cdeotte/how-to-create-tfrecords/notebook
## and here: https://www.tensorflow.org/tutorials/load_data/tfrecord
## and made changes


import tensorflow as tf
## WRITE TFRECORD ##
## adapted from here: https://www.kaggle.com/code/cdeotte/how-to-create-tfrecords/notebook
## and here: https://www.tensorflow.org/tutorials/load_data/tfrecord
## and made changes to generalize 

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class tfrec_feature(object):
    """ creates an tfrec feature object. initiate with object name. add features with add_feature function.
    based on the type of the data, it choses feature functinos from above. 
    show and show_type returns feature example and the dictionary for the data type for the feature names."""
    
    def __init__(self):
        self.feature = dict()
        self.type_dict = dict()

    def add_feature(self, data_label, data, _func_feature):
        type_data = type(data)
        self.type_dict[data_label] = type_data
        self.feature[data_label] = _func_feature(data)
               
    def show(self):
        return self.feature

    def show_type(self):
        return self.type_dict

    def serialize_example(self):
        example_proto = tf.train.Example(features=tf.train.Features(feature=self.feature))
        return example_proto.SerializeToString()




def write_TFrec_from_df_jpeg(DataFrame, iPATH_col, TFREC_structure, dict_hchy, num_dp_per_record, resize_resol,
                              labels_lookup = True, TFREC_name ="TFrec", jpeg_quality = 95):  

    """ resize_resol : list or tuple of (,)"""
    import cv2
    df = DataFrame.sample(frac=1)
    SIZE = num_dp_per_record
    config_resize = resize_resol
    table_hchy = dict_hchy
    TAR_QUALITY = jpeg_quality
    TFREC_name = TFREC_name
    format = TFREC_structure
    """ example of TFREC_strucure is as follows:
    {"image":"image", "image_id":"str", "scientificName":"int", "genus":"int", "family":"int"}
    """
    CT = len(df)//SIZE + int(len(df)%SIZE!=0)
    print(CT)
    #CT = 1
    for j in range(CT):
        print(); print('Writing TFRecord %i of %i...'%(j,CT))
        CT2 = min(SIZE,len(df)-j*SIZE)
        #CT2 = 1000
        full_name = TFREC_name+'-res%iby%i-%.2i-%i.tfrec'%(config_resize[0],config_resize[1],j,CT2)
        print(full_name)
        with tf.io.TFRecordWriter(full_name) as writer:
            for k in range(CT2):
                cidx = SIZE*j+k
                row = df.iloc[cidx]
                img = cv2.imread(row[iPATH_col])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors
                img = cv2.resize(img, config_resize)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_bytes = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, TAR_QUALITY))[1].tobytes() #tostring written by chris output [true, values], so take 1.
                _feature_ = tfrec_feature()
                for key, value in format.items():
                  if value =="image":
                    _feature_.add_feature(key, img_bytes, _bytes_feature)
                  
                  elif value =="int":
                    if labels_lookup is True:
                      _feature_.add_feature(key, table_hchy[key][row[key]], _int64_feature)  
                    else:
                      _feature_.add_feature(key, row[key], _int64_feature)
                  
                  elif value == "str":
                    _feature_.add_feature(key, bytes(row[key], 'utf-8'), _bytes_feature)
              
                example =_feature_.serialize_example()
                
                writer.write(example)
                if k%500==0: print(k,', ',end='')

### Confirm the tfrecord is correctly created ####
### tfrec_format is the same as in DataLoader ####
### tfrec_format is not the same as tfrec_structure ####
### image:str vs image:image ####

def display_sample_from_TFrec(tfrec_PATH, TFREC_FORMAT, display_size, N_suffle = 10):
    """Example of TFREC_format
    {"image":"str", "image_id":"int"}
    """
    raw_dataset = tf.data.TFRecordDataset(tfrec_PATH)
    tfrec_format = tfrec_format_generator(TFREC_FORMAT)
    parsed_dataset = raw_dataset.map(lambda x : parse_tfrecord_fn(x, tfrec_format))
    parsed_dataset = parsed_dataset.shuffle(N_suffle)
    import matplotlib.pyplot as plt

    for features in parsed_dataset.take(1):
        for key in features.keys():
            if key != "image":
                print(f"{key}: {features[key]}")

    print(f"Image shape: {features['image'].shape}")
    plt.figure(figsize=display_size)
    plt.imshow(features["image"].numpy())
    plt.show()

