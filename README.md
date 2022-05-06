[![PyPI version](https://badge.fury.io/py/mytflib.svg)](https://badge.fury.io/py/mytflib)   [![Downloads](https://pepy.tech/badge/mytflib)](https://pepy.tech/project/mytflib)

My Tensorflow Library is utility of codes for better tensorflow usage. It is written to help faster and more efficient training pipeline for big data analysis.

It consist of four main parts:

 1. DataLoader - Loading from pd.Dataframe and TFRecords, with tf.Dataset API 
 2. Learining rate Tuner and Shaper - LR searcher to find optimal range of LR, and LR shaper for scheduling different shapes of LR trajectories. 
 3. TFRecord Curator - Write, display, and manage TFRecords
 4. Training Manager - Gradient accumulation, Class re-weighting.


 ```
 pip install mytflib

 pip install git+https://github.com/johnypark/mytflib@main

 ```
 
 
