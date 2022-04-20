#### Written by
#### John Park 
#### LR Tuner 
#### 1. Find range of LR using line search
#### 2. model fit method to obtain validation and training losses 
import tensorflow as tf
import tensorflow_addons as tfa

class LRSearch(tf.keras.optimizers.schedules.LearningRateSchedule):
    
  def __init__(
      self,
      initial_learning_rate, #integer
      maximal_learning_rate,
      step_size,
      name="LRsearch"):
    
    super(LRSearch, self).__init__() #what does this do???

    self.initial_learning_rate = initial_learning_rate
    self.maximal_learning_rate = maximal_learning_rate
    self.step_size = step_size
    self.name = name

  def __call__(self, step):
     with tf.name_scope(self.name or "LRsearch") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
      step_size = tf.cast(self.step_size, dtype)
      x = tf.cast(step, dtype)
      
      return initial_learning_rate * 10 ** ( tf.experimental.numpy.log10(maximal_learning_rate/initial_learning_rate) * x / step_size)
        # this should be changed to np.exp ** (tf.log(...))


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history2 = {'loss':[],'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history2['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.history2['val_loss'].append(logs.get('val_loss'))

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_test_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )





class Search_LR():
  import tensorflow_addons as tfa

  def __init__(
      self,
      ftn_to_build_model,
      num_classes,
      resize_resol,
      optimizer: tf.keras.optimizers.Optimizer,
      loss_fn: tf.keras.losses.Loss,
      LR_range, #list
      total_steps,
      TPU_strategy,
      wd_ls = 5e-5, 
      momentum_ls = 0.9,
      name="Search_LR"
      ) -> None:
    
    super(Search_LR, self).__init__() #what does this do???
    
    self.iLR = LR_range[0]
    self.mLR = LR_range[1]
    self.get_model = ftn_to_build_model
    self.name = name
    self.spEpoch = total_steps
    self.LRschedule = LRSearch(
      initial_learning_rate = self.iLR,
      maximal_learning_rate = self.mLR,
      step_size =  self.spEpoch,
      name = 'LRSearch'
      )
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.wd = wd_ls
    self.mtm = momentum_ls
    self.strategy = TPU_strategy
    self.get_model = ftn_to_build_model
    self.num_classes = num_classes
    self.resize_resol = resize_resol

  def __call__(self, tr_ds, vali_ds, N_incre, class_weight = None, callbackclass = LossHistory()):
    import tensorflow_addons as tfa
    with self.strategy.scope():
      model = self.get_model(self.num_classes, self.resize_resol)
      model.compile(
          optimizer= self.optimizer(learning_rate = self.LRschedule, 
                                        weight_decay = self.wd, 
                                        momentum = self.mtm),
          loss = self.loss_fn,
          metrics = tfa.metrics.F1Score(num_classes = self.num_classes)
      )
      
    historylog = callbackclass

    #apply callback to override LR schedule
    if class_weight is None:
      history = model.fit(
        tr_ds, 
        epochs= N_incre,
        steps_per_epoch = int(self.spEpoch/N_incre), #Change this to round - up
        validation_data = vali_ds,
        callbacks = [historylog],
        verbose=1)
      
    else:
       history = model.fit(
        tr_ds, 
        epochs= N_incre,
        steps_per_epoch = int(self.spEpoch/N_incre), #Change this to round - up
        validation_data = vali_ds,
        class_weight = class_weight,
        callbacks = [historylog],
        verbose=1)

    
    import matplotlib.pyplot as plt
    LR_domain = range(0,int(self.spEpoch/N_incre)*N_incre)
    x = self.LRschedule(LR_domain)
    plt.plot(x, historylog.history2["loss"])
    plt.xscale('log')
    
    return history, historylog

  ## here make a plot and get the outcomes!

class Search_LR_from_compiled():
  import tensorflow_addons as tfa
  def __init__(
      self,
      compiled_model : tf.keras.Model,
      LR_range, #list
      total_steps,
      name="Search_LR"
      ) -> None:
    
    super(Search_LR, self).__init__() #what does this do???
    
    self.iLR = LR_range[0]
    self.mLR = LR_range[1]
    self.model = compiled_model
    self.name = name
    self.spEpoch = total_steps
    self.LRschedule = LRSearch(
      initial_learning_rate = self.iLR,
      maximal_learning_rate = self.mLR,
      step_size =  self.spEpoch,
      name = 'LRSearch'
      )
    
    ## Need callback

  def __call__(self, tr_ds, vali_ds, N_incre, class_weight = None, callbackclass = LossHistory()):

    historylog = callbackclass

    #apply callback to override LR schedule
    if class_weight is None:
      history = model.fit(
        tr_ds, 
        epochs= N_incre,
        steps_per_epoch = int(self.spEpoch/N_incre), #Change this to round - up
        validation_data = vali_ds,
        callbacks = [historylog],
        verbose=1)
      
    else:
       history = model.fit(
        tr_ds, 
        epochs= N_incre,
        steps_per_epoch = int(self.spEpoch/N_incre), #Change this to round - up
        validation_data = vali_ds,
        class_weight = class_weight,
        callbacks = [historylog],
        verbose=1)

    return history, historylog

  ## here make a plot and get the outcomes!
