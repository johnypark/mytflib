import numpy as np
import os
import csv

class SaveModelHistory(tf.keras.callbacks.Callback):
    #https://stackoverflow.com/questions/60727279/save-history-of-model-fit-for-different-epochs
    #modified from the above code
    
    def __init__(self,
                 outfilename, 
                 oPATH="./",
                 **kargs):
        super(SaveModelHistory,self).__init__(**kargs)
        self.OFname = outfilename
        self.oPATH = oPATH
        if (self.OFname in os.listdir(self.oPATH)):
            self.OFname = self.OFname.split(".csv")[0]+"_1"+".csv" 
        self.fPATH = os.path.join(self.oPATH, self.OFname)
        
    def on_epoch_end(self,batch,logs=None):
        if ('lr' not in logs.keys()):
            logs.setdefault('lr',0)
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

        if not (self.OFname in os.listdir(self.oPATH)):
            with open(self.fPATH,'a') as f:
                y = csv.DictWriter(f,logs.keys())
                y.writeheader()

        with open(self.fPATH,'a') as f:
            y=csv.DictWriter(f,logs.keys())
            logs_mod = dict()
            for key, value in logs.items():
                logs_mod[key] = float(np.mean(value))
            y.writerow(logs_mod)


## Class reweighting strategies for class imbalance 

def class_balanced_weight(labels, N_unique_proto):
    import numpy as np
    #Get frequency and number of classes
    ni = np.bincount(labels)
    Nclass= len(np.unique(labels))
    
    #check if noshow class presents
    class_freq = ni
    noshow_class = [i for i, freq in enumerate(class_freq) if freq==0]
    
    if len(noshow_class) > 0:
      adjusted_labels = list(labels) + noshow_class
      ni = np.bincount(adjusted_labels)
    
    #calculate class-balanced weight followed by Cui et al. "Class-balanced loss based on effective number of samples"
    Ni = N_unique_proto #given hyper parameter, as LagerN -> inf, the equqation returns compute_class_weight("balanced") from sklearn. 
    beta = (Ni-1)/(Ni)
    invEffn= (1 - beta)/(1 - beta**(ni))
    class_weights = invEffn
    min_weight = np.min(class_weights)
    
    if len(noshow_class) > 0:
      class_weights[noshow_class] = 0
      
    SumW = np.sum(class_weights)  
    class_weights = class_weights *1/min_weight
    SumW = np.sum(class_weights)
    class_weights = class_weights* Nclass / SumW
    return class_weights #reweight so sum of all weights equals to number of classes. 
  
  
def GetDictCls(GivenWeight):
    class_weight = dict()
    i = 0
    for id in range(0,len(GivenWeight)):
        class_weight[id] = round(GivenWeight[i],2)
        i = i+1
    return class_weight


def ConvertLabelsToInt(ls_labels):
  if type(ls_labels) is not list:
    print("error!")
  Ordered = sorted(set(ls_labels))
  i = 0
  LookUp = dict()
  for label in Ordered:
    LookUp[label] = i
    i += 1

  RevLookUp = dict()
  ls_labels_int = [LookUp[item] for item in ls_labels]
  RevLookUp = zip(ls_labels_int, ls_labels)
  return ls_labels_int, dict(RevLookUp)

##############################################
#### FOCAL LOSS from TensorFlow Addons #####
############################################
#### FIX : 1) Complex Number Generation by clipping focal_weight. 2) set reduction to AUTO mitigates exploding gradient.
####
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements Focal loss."""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

def sigmoid_focal_crossentropy(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.25,
    gamma: FloatTensorLike = 2.0,
    from_logits: bool = False
    ):
  
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, 
                               from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1.0 - y_true) * (1.0 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0
    
    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        focal_weight = K.clip((1.0 - p_t), K.epsilon(), 1.0) 
        
        ### The change from tfa_focal_loss : clipped focal_weight to elemniate  
        ### cases that focal_weight throws negative value, in which case induces NaN values of the loss
        ### for gammma <1 case.
        modulating_factor = tf.pow(focal_weight, gamma)

    Result = tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
    #Result = tf.nn.relu(Result)
    # compute the final loss and return
    return Result

class SigmoidFocalCrossEntropy(LossFunctionWrapper):
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much higher for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    """
    def __init__(
        self,
        from_logits: bool = False,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        
        name: str = "sigmoid_focal_crossentropy",
    ):
        super().__init__(
            sigmoid_focal_crossentropy, 
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma
        )

