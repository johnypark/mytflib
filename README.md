[![PyPI version](https://badge.fury.io/py/mytflib.svg)](https://badge.fury.io/py/mytflib)   [![Downloads](https://pepy.tech/badge/mytflib)](https://pepy.tech/project/mytflib)

My Tensorflow Library contains utility codes for better TensorFlow usage. Tensorflow is a great tool, but it is not the best when it comes to convenience of usage. My TensorFlow Library is built to fill this gap. This package is especially useful if you are struggling with large training dataset with imbalanced classes.  It will enable faster and more efficient training pipeline for big data analysis. 

It consists of four main parts:

 1. DataLoader - Loading from pd.Dataframe and TFRecords, with tf.Dataset API 
 2. Learining rate Tuner and Shaper - LR searcher to find optimal range of LR, and LR shaper for scheduling different shapes of LR trajectories (consine, linear, exponential, tangent, warmup, ...). 
 3. TFRecord Curator - Write, display, and manage TFRecords
 4. Training Manager - Gradient accumulation, Class re-weighting.


 ```
 pip install mytflib

 pip install git+https://github.com/johnypark/mytflib@main

 ```
 
 
