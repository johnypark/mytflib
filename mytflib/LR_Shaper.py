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

import tensorflow as tf

class CyclicalExpLR(tf.keras.optimizers.schedules.LearningRateSchedule):
      
  #@typechecked
  def __init__(
    self,
    initial_learning_rate, #3: Union[FloatTensorLike, Callable],
    maximal_learning_rate, #3: Union[FloatTensorLike, Callable],
    step_size, #3: FloatTensorLike,
    scale_fn = lambda x:1, #3: Callable,
    scale_fn_min = lambda x:1,
    scale_mode = "cycle", #3: str = "cycle",
    name ="CycicalExpLR" #3: str = "CyclicalLearningRate",
    ):
      super().__init__()
      self.initial_learning_rate = initial_learning_rate
      self.maximal_learning_rate = maximal_learning_rate
      self.step_size = step_size
      self.scale_fn = scale_fn
      self.scale_fn_min = scale_fn_min
      self.scale_mode = scale_mode
      self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "CyclicalLearningRate"):
      initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
      dtype = initial_learning_rate.dtype
      maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
      step_size = tf.cast(self.step_size, dtype)
      step_as_dtype = tf.cast(step, dtype)
      cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
      x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)

      mode_step = cycle if self.scale_mode == "cycle" else step

      #return initial_learning_rate + (maximal_learning_rate - 
      #        initial_learning_rate) * tf.maximum(tf.cast(0, dtype), (1 - x)) * self.scale_fn(mode_step)

      result = initial_learning_rate * np.e ** ( tf.maximum(tf.cast(0, dtype), (1 - x)) * self.scale_fn(mode_step) * tf.math.log (maximal_learning_rate / initial_learning_rate
              )) 

      return result

  def get_config(self):
    return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "scale_fn": self.scale_fn,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }

from numpy.core.numeric import base_repr

class piecewise_ftn():
    
    
  def __init__(self, config, LineType = ["linear", "exponential","tangent","exponential tangent"][0]):
    self.config = config
    #CurveType has "linear", "exponential", "tan", "cos"...
    self.LineType = LineType
    #potentially calculate exponential case for all functions... 

    if type(config) is dict:
      self.INIT_STEP = self.config['INIT_STEP'] 
      self.STEP_SIZE = self.config['STEP_SIZE']
      self.INIT_LR = self.config['INIT_LR']
      self.MAX_LR = self.config['MAX_LR']

    elif type(config) is list:
      self.INIT_STEP= self.config[0]
      self.STEP_SIZE = self.config[1]
      self.INIT_LR = self.config[2]
      self.MAX_LR = self.config[3]
      # warn if len(config)!=4

    # When using equations with lambda funcion:
    #if self.LineType == "linear":
    #  eq = "(self.MAX_LR-self.INIT_LR)/(self.STEP_SIZE)*({}) + self.INIT_LR".format("x-self.INIT_STEP")
    #elif self.LineType == "exponential":
    #  eq = "y1* np.e ** ( np.log(y2/y1) *(x - x1)/(x2 - x1))"
    # elif self.cType == "tangent":
    #self.eq = eq

    print("Equation:{} for x1={}, x2={}, y1={}, y2={}".format(self.LineType, 
                                self.INIT_STEP,self.INIT_STEP + self.STEP_SIZE, 
                                self.INIT_LR, self.MAX_LR)) 
  def __call__(self,x,tanx_max = 3):
    x = np.array(x)
    #print("Calculating {} for {}".format(self.eq,x))
    SIGN = 1
    x = x - self.INIT_STEP

    if "tangent" in self.LineType and self.INIT_LR > self.MAX_LR:
      K = tanx_max
      x = np.pi - ( x + np.arctan(K)) # this setup gives convex curve fatter than cosine.
      #x = 2*np.arctan(K)*self.STEP_SIZE - x
      SIGN = (-1)* SIGN

    if self.LineType == "linear":
      fx = (self.MAX_LR-self.INIT_LR)/(self.STEP_SIZE)*(x) + self.INIT_LR

    elif self.LineType == "exponential":
      fx = self.INIT_LR* np.e ** ( np.log(self.MAX_LR/self.INIT_LR) *(x)/(self.STEP_SIZE))
    
    elif self.LineType == "tangent":
      K = tanx_max
      fx = SIGN*(self.MAX_LR-self.INIT_LR)/K*np.tan((x/self.STEP_SIZE)*(np.arctan(K)))+self.INIT_LR

    elif self.LineType == "exponential tangent":
      K = tanx_max
      fx = self.INIT_LR * np.e **(SIGN * np.log(self.MAX_LR/self.INIT_LR)/K*np.tan((x/self.STEP_SIZE)*(np.arctan(K))))

    return fx
  ##Method plot plot results

  def get_config(self):
    return {
            "initial_learning_rate": self.INIT_LR,
            "maximal_learning_rate": self.MAX_LR,
            #"scale_fn": self.scale_fn,
            "step_size": self.STEP_SIZE,
            #"scale_mode": self.scale_mode,
        }