import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os 
import glob
import random
import cv2
# this file is for TFREC writer function.



def h22_getfPATH(rPATH, df, Dtype="train"):
    if Dtype != "train":
        print("error")
    ls_fPATHs = [0]*len(df)
    pos_split = 3
    for i in range(len(df)):
        c_df = df.iloc[i]
        zf_cat_id = str(c_df.category_id).zfill(5)
        ls_fPATHs[i] =  os.path.join(rPATH, zf_cat_id[:pos_split], zf_cat_id[pos_split:], c_df.image_id+".jpg")
    return ls_fPATHs


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

    def add_feature(self, data_label, data):
        type_data = type(data)
        self.type_dict[data_label] = type_data
        
        if type_data is bytes:
          _func_feature = _bytes_feature
        elif type_data is int:
          _func_feature = _int64_feature
        elif type_data is float:
          _func_feature = _float_feature
        
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
        full_name = TFREC_name+'-res%i-train%.2i-%i.tfrec'%(config_resize,j,CT2)
        print(full_name)
        with tf.io.TFRecordWriter(full_name) as writer:
            for k in range(CT2):
                cidx = SIZE*j+k
                row = df.iloc[cidx]
                img = cv2.imread(row[iPATH_col])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors
                img = cv2.resize(img, [config_resize, config_resize])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_bytes = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, TAR_QUALITY))[1].tobytes() #tostring written by chris output [true, values], so take 1.
                _feature_ = tfrec_feature()
                for key, value in format.items():
                  if value =="image":
                    _feature_.add_feature(key, img_bytes)
                  
                  elif value =="int":
                    if labels_lookup is True:
                      _feature_.add_feature(key, table_hchy[key][row[key]])  
                    else:
                      _feature_.add_feature(key, row[key])
                  
                  elif value == "str":
                    _feature_.add_feature(key, bytes(row[key], 'utf-8'))

                example =_feature_.serialize_example()
                
                writer.write(example)
                if k%1000==0: print(k,', ',end='')



