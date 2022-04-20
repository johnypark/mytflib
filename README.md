
My Tensorflow Library is collection of codes for a more fluent tensorflow usage. 

It consist of four main parts:

 1. DataLoader - Loading from pd.Dataframe and TFRecords, with tf.Dataset API 
 2. Learining rate Tuner and Shaper - LR searcher after Smith (2018) to find optimal range of LR, various shape of LR curves.
 3. TFRecord Curator - Write, display, and manage TFRecords
 4. Training Manager - Gradient accumulation, Class re-weighting.


 ```
 pip install mytflib==0.0.1.6

 pip install git+https://github.com/johnypark/mytflib@main

 ```
 
 