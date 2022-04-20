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

## use DataLoader.tfrec_format_generator to generate features. from dictionary?
def serialize_example(f_image, f_image_id, f_cat_id, f_genus, f_family):
  feature = {
      'image': _bytes_feature(f_image),
      'image_id': _bytes_feature(f_image_id),
      'category_id': _int64_feature(f_cat_id),
      'genus': _bytes_feature(f_genus),
      'family': _bytes_feature(f_family)
    }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def write_TFrec_from_df_jpeg(DataFrame, num_dp_per_record, resize_resol, jpeg_quality = 95):  

    SIZE = num_dp_per_record # Specify # of images that goes into one tfrecord file
    RESIZE = resize_resol
    TAR_QUALITY = jpeg_quality # 
    df = DataFrame.sample(frac=1)
    ### row.fPATH: file PATH 
    ### row.image_id
    ### row.genus
    ### row.family

    CT = len(df)//SIZE + int(len(df)%SIZE!=0)
    #CT = 1
    for j in range(CT):
        print(); print('Writing TFRecord %i of %i...'%(j,CT))
        CT2 = min(SIZE,len(df)-j*SIZE)
        #CT2 = 1000
        with tf.io.TFRecordWriter('h22-miniv1-res%i-train%.2i-%i.tfrec'%(RESIZE[0],j,CT2)) as writer:
            for k in range(CT2):
                cidx = SIZE*j+k
                row = df.iloc[cidx]
                img = cv2.imread(row.fPATH)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors
                img = cv2.resize(img, RESIZE)
                img_bytes = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, TAR_QUALITY))[1].tobytes() #tostring written by chris output [true, values], so take 1.
                example = serialize_example(
                    img_bytes, 
                    tf.io.serialize_tensor(row.image_id), #tensorflow string utf8 
                    #https://www.tensorflow.org/text/api_docs/python/text/normalize_utf8 may be a solution
                    row.category_id,
                    tf.io.serialize_tensor(row.genus),
                    tf.io.serialize_tensor(row.family)
                )

                writer.write(example)
                if k%100==0: print(k,', ',end='')