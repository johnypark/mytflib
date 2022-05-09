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

import os
import csv
import numpy as np
from timeit import default_timer as timer
import re
import json
import tensorflow as tf
from contextlib import redirect_stdout


def change_np_float_to_float(Dict):
    
  for key,value in Dict.items():
    if type(value) not in [str, float, bool]:
      if type(value) is dict:
        change_np_float_to_float(Dict[key]) ##Recursive in case of nested dictionary
      elif value is not None:
        Dict[key] = float(value) 
  return(Dict)


def model_config_save(model,
                      config_info, 
                      json_file_name, 
                      txt_file_name,
                      oPATH):

  model_info = dict()
  model_info['optimizer'] = model.optimizer.get_config()
    
  try:
    model_info['loss'] = model.loss.get_config()
  
  except:
    model_info['loss'] = dict()
    i = 0;
    for loss_i in model.loss:
      model_info['loss'][i] = loss_i.get_config()
      i +=1
        
  model_info['optimizer'] = change_np_float_to_float(model_info['optimizer'])
  model_info['loss'] = change_np_float_to_float(model_info['loss'])
  print("model optimizer loss type adjusted for json serialization")
  model_info['config_info'] = config_info
  print("grab config info")
  fPATH = os.path.join(oPATH,json_file_name)
  with open(fPATH, 'w') as outfile:
    json.dump(model_info, outfile, sort_keys = True, indent = 4,
               ensure_ascii = False)
  print("done - saving config info to {}".format(fPATH))
  with open(os.path.join(oPATH, txt_file_name), 'w') as f:  #make this automatic, within SaveModelHistory
    with redirect_stdout(f):
        model.summary()
  print("model summary saved to {}. initialization is done".format(txt_file_name))
        

class SaveModelHistory(tf.keras.callbacks.Callback):
    #https://stackoverflow.com/questions/60727279/save-history-of-model-fit-for-different-epochs
    #modified from the above code
    def __init__(self,
                 config_info,
                 outfilename, 
                 oPATH="./",
                 **kargs):
        super(SaveModelHistory,self).__init__(**kargs)
        
        self.config_info = config_info
        self.OFname = outfilename.split(".")[0]+".csv"
        self.oPATH = oPATH

        if (self.OFname in os.listdir(self.oPATH)):
          print("{} already exsits".format(self.OFname))
          NameExistError = True
          self.OFname =  self.OFname.split(".csv")[0]+"_1.csv" 
          
          while NameExistError:
            if (self.OFname in os.listdir(self.oPATH)):
              print("{} already exsits".format(self.OFname))
              raw_name = self.OFname.split(".csv")[0].split("_")[:-1]
              name_index = re.findall(r'\_\d+\b', self.OFname)[0].split("_")[1]
              name_index = str(int(name_index) + 1)
              self.OFname ="_".join(raw_name+[name_index])+".csv"
            else:
              print("using filename {} for saving current training task.".format(self.OFname))
              NameExistError = False
              
        self.fPATH = os.path.join(self.oPATH, self.OFname)
        
        #Adapted from: https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        
    def on_train_begin(self, logs = None):
        
        json_file_name = self.OFname.split(".csv")[0]+"_configs.json"
        model_summary_file_name = self.OFname.split(".csv")[0]+"_model_summary.txt" 
        
        model_config_save(model = self.model, config_info = self.config_info, 
                          json_file_name = json_file_name, 
                          txt_file_name = model_summary_file_name,
                          oPATH = self.oPATH)
        
        self.model_optim_config = self.model.optimizer.get_config()
        print(self.model_optim_config)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = timer()
        #print("epoch begun - timer set")
        
    def on_epoch_end(self, epoch, logs=None):
        extra_logs = dict()
        extra_logs['learning_rate'] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        
        extra_logs['weight_decay'] = tf.keras.backend.get_value(self.model.optimizer.weight_decay)
        
        try:
          extra_logs['momentum'] = tf.keras.backend.get_value(self.model.optimizer.momentum)
        except:
          no_momentum =True
        
        for hyperparam, value in extra_logs.items():
          if type(self.model_optim_config[hyperparam]) is dict:
            #print(extra_logs[hyperparam])
            extra_logs[hyperparam] = float(value((epoch+1) * self.config_info["steps_per_epoch"])) ## here replace with self.params.get('steps')
        
        logs['time'] = timer()-self.start_time
        
        #print("epoch end - dict lr, momentum, wd, time")
        #print(extra_logs, logs)

        if not (self.OFname in os.listdir(self.oPATH)):
            with open(self.fPATH,'a') as f:
                y = csv.DictWriter(f,list(logs.keys())+list(extra_logs.keys()))
                y.writeheader()

        with open(self.fPATH,'a') as f:
            y=csv.DictWriter(f,list(logs.keys())+list(extra_logs.keys()))
            logs_mod = dict()
            for key, value in logs.items():
                logs_mod[key] = float(np.mean(value))
            for key, value in extra_logs.items():
                logs_mod[key] = float(np.mean(value))
            
            y.writerow(logs_mod)
    
        #print("epoch end - successfully appeneded lr, momentum, wd, time to {}".format(self.fPATH))

## Class reweighting strategies for class imbalance 


def class_balanced_weight(labels, max_range):
    """
    Args:
      labels (list): list of integer labels for the entire dataset. e.g., labels = [0, 1, 0, 1, 2, 0, 1, 0, 0, 0] for dataset with len(ds)=10 and N_class = 3
      max_range (int): max_range of weight variation from the best represented class to the least represented class. if max_range=inf, then weight equals \
                       to the inverse class frequency

    Returns:
      _type_: list of class weights indexed by integer number of each class. They are reweighted that sums of the weights equalts to N_class, \
        so it could yield comparable loss value to no weight case. 
    """
  
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
    Ni = max_range #given hyper parameter, as LagerN -> inf, the equqation returns compute_class_weight("balanced") from sklearn. 
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
    print("error! input is not a list")
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