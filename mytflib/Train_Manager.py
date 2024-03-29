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


# 5/21/22 
class EpochBasedScheduler:
  def __init__(self, LR_ftn_by_step, steps_per_epoch):
    self.LR_ftn_by_step = LR_ftn_by_step
    self.steps_per_epoch = steps_per_epoch

  def __call__(self, epoch):
    return self.LR_ftn_by_step(epoch * self.steps_per_epoch)


#https://stackoverflow.com/questions/65905991/learning-rate-callback-on-step-rather-than-epoch
class UpdateLR_byStep(tf.keras.callbacks.Callback):
    def __init__(self, schedule, steps_per_epoch, pickup=None):
        super(UpdateLR_byStep, self).__init__()
        self.schedule = schedule
        self.epoch = 0
        self.steps_per_epoch = steps_per_epoch
        if pickup:
          self.epoch = pickup
        
    def on_batch_begin(self, batch, logs=None):
        actual_step = (self.epoch*self.steps_per_epoch) + batch
        scheduled_lr = self.schedule(actual_step)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        if batch == 0:
            print("--Learning Rate: {:.3g} --".format(scheduled_lr))
        self.scheduled_lr = scheduled_lr
        
    def on_epoch_end(self, epoch, logs=None):
        #message = '  LR: {}.'.format(self.scheduled_lr)
        #sys.stdout.write(message + "\n")
        self.epoch+=1


def change_np_float_to_float(Dict):
    
  for key, value in Dict.items():
    if type(value) not in [str, float, bool]:

      if type(value) is dict:
        change_np_float_to_float(Dict[key]) ##Recursive in case of nested dictionary

      #elif type(value) is type(tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
    #tf.constant([1,1]), tf.constant([1,1])),default_value = 0)):

    
        #Dict[key] = float(value.size().numpy())

      elif type(value) is list:
        Dict[key] =[float(ele) for ele in value]
          
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
      try:
        model_info['loss'][i] = loss_i.get_config()
      except:
        model_info['loss'][i] = model.loss[loss_i].get_config()
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
  
  
def get_output_path(out_file_name,outPATH, extension = ".csv"):
    ext = extension
    OFname = out_file_name.split(ext)[0]+ext
    oPATH = outPATH
    if (OFname in os.listdir(oPATH)):
        print("{} already exsits".format(OFname))
        NameExistError = True
        OFname =  OFname.split(ext)[0]+"_1"+ext 
            
        while NameExistError:
            if (OFname in os.listdir(oPATH)):
                print("{} already exsits".format(OFname))
                raw_name = OFname.split(ext)[0].split("_")[:-1]
                name_index = re.findall(r'\_\d+\b', OFname)[0].split("_")[1]
                name_index = str(int(name_index) + 1)
                OFname ="_".join(raw_name+[name_index])+ext
            else:
                print("using filename {} for saving current training task.".format(OFname))
                NameExistError = False
                
    fPATH = os.path.join(oPATH, OFname)
        
    return(fPATH)
        

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
        self.fPATH = get_output_path(out_file_name = self.OFname, 
                                     outPATH = self.oPATH,
                                     extension = ".csv")
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
    
    Citation:
    "Class-Balanced Loss Based on Effective Number of Samples", by Yin Cui et al. 2019. CVPR.
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
    
  #if type(ls_labels) is not list:
  #  print("error! input is not a list")
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

  
### PatristicLoss written by Dr. Damon Little
class PatristicLoss(tf.keras.losses.Loss):
	def __init__(self, 
		batch_size, 
		g2d = None, 
		t2g = None,
    name='Patristic_Distance'):
		super().__init__()
		self.genera2distance = g2d
		self.taxonID2genus = t2g
		self.batch = batch_size

	def call(self, y_true, y_pred):
		# weights = tf.ones((settings['batch'], 1), dtype = settings['dtype'])
		# if 'sample_weight' in kwargs:
		# 	weights = kwargs.get('sample_weight')
		#
		# apply mix to distances
		#
		#
		# faster to use a 'static' tensor and tf.gather?
		#
		trueGenera = self.taxonID2genus.lookup(tf.argmax(y_true , axis = 1, output_type = tf.int32)) #self.taxonID2genus.lookup(tf.cond(
			#tf.equal(tf.shape(y_true)[1], 4),
			#lambda: tf.reshape(tf.gather(y_true, [0], axis = 1), [-1]), ### true; training; major taxonID
			#lambda: tf.reshape(y_true, [-1]) ### false; testing; taxonID
		#))
		predictedGenera = self.taxonID2genus.lookup(tf.argmax(y_pred , axis = 1, output_type = tf.int32))
		pdist = tf.reshape(tf.math.maximum(
			self.genera2distance.lookup(tf.strings.reduce_join(
				axis = -1,
				inputs = tf.stack([trueGenera, predictedGenera], axis = 1),
				separator = ' <=> '
			)),
			self.genera2distance.lookup(tf.strings.reduce_join(
				axis = -1,
				inputs = tf.stack([predictedGenera, trueGenera], axis = 1),
				separator = ' <=> '
			))
		), (self.batch, 1))
		# return tf.constant(0.0)
		# return tf.zeros((settings['batch'], 1))
		# tf.print(tf.math.reduce_sum(pdist))
		return pdist #tf.math.reduce_sum(pdist) ### works but includes weights?
		# return pdist
		# return tf.math.reduce_sum(tf.math.multiply(pdist, weights))


def get_genera2distance(DistanceFilename, data_type):	
	distance = dict()### 'Genus <=> Genus' => distance
	with open(DistanceFilename, mode = 'rt') as file:
		for k, line in enumerate(file):
			if k > 0:
				columns = line.rstrip().split('\t')
				distance[f"{columns[0]} <=> {columns[1]}"] = float(columns[2])
	genera2distance = tf.lookup.StaticHashTable( ### 'Genus <=> Genus' => distance
		tf.lookup.KeyValueTensorInitializer(
			tf.constant(list(distance.keys()), dtype = tf.string), 
			tf.constant(list(distance.values()), dtype = data_type)
			),
		default_value = 0.0
		)
	return genera2distance


def get_class2genus(ls_class, ls_genus):
    class2genus = tf.lookup.StaticHashTable( ### taxonID => Genus
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(ls_class), dtype = tf.int32), 
            tf.constant(list(ls_genus), dtype = tf.string)
            ),
            default_value = 'NOT A GENUS'
        )

    return class2genus


def get_taxonID2genus_from_df(df_PATH, keycol, valcol):
    import pandas as pd
    df = pd.read_table(df_PATH)
    S_scif = sorted(set(df[keycol]))
    mapping_scif2int = dict(zip(S_scif, range(len(S_scif))))
    df_by_scif = df.groupby(keycol, as_index= False).first() 
    ls_class = [mapping_scif2int[ele] for ele in df_by_scif[keycol]]
    ls_genus = [ele for ele in df_by_scif[valcol]]
    return get_class2genus(ls_class, ls_genus)

def categorical_crossentropy(target, output, from_logits=False, axis=-1):
  """Categorical crossentropy between an output tensor and a target tensor.
  Args:
      target: A tensor of the same shape as `output`.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last`, and `axis=1` corresponds to data format
          `channels_first`.
  Returns:
      Output tensor.
  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.
  Example:
  >>> a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
  >>> print(a)
  tf.Tensor(
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]], shape=(3, 3), dtype=float32)
  >>> b = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94], shape=[3,3])
  >>> print(b)
  tf.Tensor(
    [[0.9  0.05 0.05]
     [0.05 0.89 0.06]
     [0.05 0.01 0.94]], shape=(3, 3), dtype=float32)
  >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
  >>> print(np.around(loss, 5))
  [0.10536 0.11653 0.06188]
  >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
  >>> print(np.around(loss, 5))
  [0. 0. 0.]
  """
  target = tf.convert_to_tensor(target)
  output = tf.convert_to_tensor(output)
  target.shape.assert_is_compatible_with(output.shape)

  # Use logits whenever they are available. `softmax` and `sigmoid`
  # activations cache logits on the `output` Tensor.
  if hasattr(output, '_keras_logits'):
    output = output._keras_logits  # pylint: disable=protected-access
    if from_logits:
      warnings.warn(
          '"`categorical_crossentropy` received `from_logits=True`, but '
          'the `output` argument was produced by a sigmoid or softmax '
          'activation and thus does not represent logits. Was this intended?"',
          stacklevel=2)
    from_logits = True

  if from_logits:
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=target, logits=output, axis=axis)

  if (not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable)) and
      output.op.type == 'Softmax') and not hasattr(output, '_keras_history'):
    # When softmax activation function is used for output operation, we
    # use logits from the softmax function directly to compute loss in order
    # to prevent collapsing zero when training.
    # See b/117284466
    assert len(output.op.inputs) == 1
    output = output.op.inputs[0]
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=target, logits=output, axis=axis)

  # scale preds so that the class probas of each sample sum to 1
  output = output / tf.reduce_sum(output, axis, True)
  # Compute cross entropy from probabilities.
  epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
  output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)
  return -tf.reduce_sum(target * tf.math.log(output), axis)
