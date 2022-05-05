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



#!/usr/bin/env python
# adpoted from https://github.com/Dtananaev/one_cycle_scheduler_tf
"""
Copyright (c) 2021 Tananaev Denis
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
import numpy as np


class OneCycle(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A OneCycle that uses an cosine annealing schedule for cycle."""

    def __init__(
        self,
        initial_learning_rate,
        maximal_learning_rate,
        cycle_size,
        scale_fn=lambda x: 1.0,
        shift_peak=0.3,
        scale_mode="cycle",
        final_lr_scale=1.0,
        name=None,
    ):
        """
        Copyright (c) 2021 Tananaev Denis
        Applies cyclical cosine annealing learning rate.
        It is possible to get the same learning rate scheduler as it was
        used by FastAI for superconvergence: https://docs.fast.ai/callbacks.one_cycle.html
        or Kaggle post: https://www.kaggle.com/avanwyk/tf2-super-convergence-with-the-1cycle-policy
        In order to do that:
        ```python
         maximal_learning_rate = <value from lr finder>
        initial_learning_rate = maximal_learning_rate / 25.0
        cycle_size = 3-5 epoch (should be defined by you) It defines size of cycle
        lr_schedule = CyclicalCosineAnnealing(
            initial_learning_rate,
            maximal_learning_rate,
            cycle_size,
            scale_fn = lambda x: 1.0, or lambda x: tf.where(x > 1.0, 0.8, 1.0)
            shift_peak = 0.3
            scale_mode="cycle",
           final_lr_scale=1e-4)
        ```
        The learning rate schedule is also serializable and deserializable using
        `tf.keras.optimizers.schedules.serialize` and
        `tf.keras.optimizers.schedules.deserialize`.
        Args:
        initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
        maximal_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The maximal learning rate.
        step_size: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the  half cycle size in interations.
        scale_fn: scale your cycle (make it bigger/smaller for the next cycle)
        shift_peak: shift the pick point to the left side
        scale_mode: scale by "cycle" or "step"
        final_lr_scale: filal_lr = initial_learning_rate * final_lr_scale
        name: String.  Optional name of the operation.
        Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the cyclical learning rate, a scalar `Tensor` of the same
        type as `initial_learning_rate`.
        """
        super(OneCycle, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.cycle_size = cycle_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.shift_peak = shift_peak
        self.final_lr_scale = final_lr_scale
        self.name = name
        # Defines the position of the max lr in steps
        self._total_steps = cycle_size
        self._first_half_steps = shift_peak * self._total_steps
        self._second_half_steps = self._total_steps - self._first_half_steps

    def get_cosine_annealing(self, start, end, step, step_size_part, cycle):
        x = step / step_size_part
        cosine_annealing = 1 + tf.math.cos(tf.constant(np.pi) * x)
        return end + 0.5 * (start - end) * cosine_annealing

    def __call__(self, step, optimizer=False):
        with tf.name_scope(self.name or "OneCycle"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            # Cast all internal members to necessary type
            step = tf.cast(step, dtype)
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            first_half_steps = tf.cast(self._first_half_steps, dtype)
            second_half_steps = tf.cast(self._second_half_steps, dtype)
            total_steps = tf.cast(self._total_steps, dtype)
            final_lr_scale = tf.cast(self.final_lr_scale, dtype)
            # Check in % the cycle
            cycle_progress = step / total_steps
            cycle = tf.floor(1 + cycle_progress)

            percentage_complete = 1.0 - tf.abs(cycle - cycle_progress)  # percent of iterations done
            first_half = tf.cast(percentage_complete <= self.shift_peak, dtype)

            normalized_first_half_step = step - (cycle - 1) * total_steps
            normalized_second_half_step = normalized_first_half_step - first_half_steps
            final_lr = initial_learning_rate * final_lr_scale

            lr_begin = self.get_cosine_annealing(
                initial_learning_rate,
                maximal_learning_rate,
                normalized_first_half_step,
                first_half_steps,
                cycle,
            )
            lr_end = self.get_cosine_annealing(
                maximal_learning_rate,
                final_lr,
                normalized_second_half_step,
                second_half_steps,
                cycle,
            )

            lr_res = first_half * lr_begin + (1.0 - first_half) * lr_end
            mode_step = cycle if self.scale_mode == "cycle" else step

            if optimizer == False:
                lr_res = lr_res * self.scale_fn(mode_step)

            return lr_res

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "cycle_size": self.cycle_size,
            "scale_mode": self.scale_mode,
            "shift_peak": self.shift_peak,
        }

