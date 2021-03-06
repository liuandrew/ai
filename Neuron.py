# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:49:47 2018

@author: Andy
"""

import logging
import threading
import numpy as np
import matplotlib
from random import random

import matplotlib.pyplot as plt
from matplotlib import animation

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s')
#%%
class Clock(threading.Thread):
  '''
  Internal threaded clock for neuron to time calculations against.
  Each neuron should have an instance of Clock that runs it's step function
  '''
  def __init__(self, timestep, stop_flag, target):
    threading.Thread.__init__(self)
    self.target = target
    self.timestep = timestep
    self.stopped = stop_flag
  
  def run(self):
    '''
    Check if the given stop_flag event has been set. If not, wait for the
    defined timestep and run the event the thread has been set with
    '''
    while not self.stopped.wait(self.timestep):
      self.target()

#%%
class Neuron:
  # all of these parameters will have to deal with scaling in the future
  # may be best to do every one scaled in terms of timestep
  def __init__(self, timestep=1.0, active_memory_length=10.0, name='neuron',
      plot_potential=False, activate=None, activate_threshold=0.5,
      cost_function_window=1.0, delta_cost_threshold=0.7, base_certainty=0.0001,
      annealing_steps=5, annealing_threshold=0.05, sensitizing_threshold=0.7,
      sensitizing_rate=0.02):
    '''
    Construct a new Neuron
    :param timestep: float, optional (default=1.0)
      How often the neuron processes it's time in seconds.
      this defines how often it can fire as well
    :param active_memory_length: float, optional (default=10.0)
      The length of time in seconds the neuron holds in active memory
    :param name: string, optional (default='neuron')
    :param activate: function, required
      The function to be called if the neuron decides to activate
      => need a way to hook this to another neuron
    :param activate_threshold: float, optional (default=0.5)
      The value needed to activate - higher = harder to activate
      -> importantly, this shouldn't matter too much after learning?
    :param cost_function_window: float, optional (default=1.0)
      The length of window by which cost function is calculated
      This is related to how long in the future the neuron would like to
        see if its actions had an impact on (maybe)
    :param delta_cost_threshold: float optional (default = 0.7)
      The amount of cost change needed to trigger memorization
    :param base_certainty: float optional (default = 0.001)
      The base certainty applied to each memory, i.e. how quickly the neuron
      learns from experiences. Too high and it will fall into local optima
    :param annealing_steps: int, optional (default = 5)
      The number of time steps that memories should be compared to and annealed
      This may be better represented as a decimal of total steps in active
        memory
    :param annealing_threshold: float, optional (default = 0.05)
      Threshold of distance below which a memory should be considered a match to
      anneal
    :param sensitizing_threshold: float, optional (default = 0.7)
      Threshold below which the neuron will sensitize itself
    :param sensitizing_rate: float, optional (default = 0.02)
      The maximum rate at which the neuron will sensitize itself, as a fraction
      of current sensitivity
    '''
    # neuron state variables
    self.memories = None
    self.active_memory = np.array([])
    self.certainty_memory = np.array([])
    self.input_potential = 0
    self.sensitivity = 1.0

    # neuron parameters
    self.active_memory_length = active_memory_length
    self.active_memory_steps = int(active_memory_length / timestep)
    self.activate = activate
    self.activate_threshold = activate_threshold
    self.cost_function_window = cost_function_window
    self.cost_function_steps = int(cost_function_window / timestep)
    self.cost_steps_count = 0
    self.delta_cost_threshold = delta_cost_threshold
    self.base_certainty = base_certainty
    self.annealing_steps = annealing_steps
    self.annealing_threshold = annealing_threshold
    self.annealed_memories = None
    self.sensitizing_rate = sensitizing_rate
    self.sensitizing_threshold = sensitizing_threshold

    # internal clock
    self.stopFlag = threading.Event()
    self.clock = Clock(timestep, self.stopFlag, self.step)
    self.clock.setName(name)

    self.plot_potential = plot_potential
    if(self.plot_potential):
      self.init_potential_graph()

    # used to calculate rolling cost function -> should converge over time
    self.total_cost = 0
    self.total_cost_steps = 0
    
  def calculate(self):
    pass
  
  def step(self):
    '''
    Calculation function called at every timestep
      - Process any inputs that have been received
      - Process any patterns that are running
      - Calculate probability of firing and determine output
      - Update active memory
    '''
    # logging.debug('input_potential: {}'.format(self.input_potential))
    # logging.debug('sensitivity: {}'.format(self.sensitivity))
    # logging.debug('sensitized potential: {}'.format(self.input_potential * self.sensitivity))

    self.active_memory = np.append(self.active_memory, 
      self.input_potential * self.sensitivity)
    if(self.active_memory.size > self.active_memory_steps):
      self.active_memory = self.active_memory[1:]

    # --------------------
    # Match Memories
    # --------------------
    self.match_memories()

    # --------------------
    # Calculate Decision
    # --------------------
    certainty = self.decision_probability()

    self.certainty_memory = np.append(self.certainty_memory, certainty)
    if(self.certainty_memory.size > self.active_memory_steps):
      self.certainty_memory = self.certainty_memory[1:]

    # --------------------
    # Calculate Cost Function
    # --------------------
    self.calculate_cost_function()

    # logging.debug(self.active_memory)
    self.input_potential = 0

    if(self.plot_potential):
      self.animate_potential()


  def calculate_cost_function(self):
    '''
    For every n = self.cost_function_steps steps, calculate the last n steps of
    cost function and compare to the last [2n, n] steps of cost function
    minimize: j2 - j1 negative is a decrease in cost, this is positive influence
      on actions taken
    maximize: j2 - j1 positive is an increase in cost, this is positive on
      actions taken
    '''
    cost_function = 'minimize'

    self.cost_steps_count += 1

    if(self.cost_steps_count >= self.cost_function_steps):
      self.cost_steps_count = 0
      steps = self.cost_function_steps
      j1 = self.active_memory[-steps:]
      j2 = self.active_memory[-(2 * steps):-steps]
      deltaj = np.sum(j2) - np.sum(j1)

      j1_sum = np.sum(j1)
      logging.debug(j1_sum)
      # desensitize if j1_sum exceeds 1.0
      if(j1_sum > 1.0):
        self.sensitivity = self.sensitivity * (1.0 / j1_sum)
        # calibrate deltaj against new sensitivity
        deltaj = deltaj / j1_sum

      # sensitize if j1_sum is below sensitizing threshold
      if(j1_sum < self.sensitizing_threshold):
        # sensitize up to at maximum the threshold
        if((not j1_sum == 0) and 
          ((self.sensitizing_threshold / j1_sum) < self.sensitizing_rate)):
          self.sensitivity = self.sensitivity * (self.sensitizing_threshold / j1_sum)
        # or up to the sensitizing_rate
        else:
          self.sensitivity = self.sensitivity * (1 + self.sensitizing_rate)

      if(cost_function is 'minimize'):
        deltaj = -deltaj
      if(cost_function is 'maximize'):
        pass
      if(abs(deltaj) > self.delta_cost_threshold):
        self.memorize(deltaj)

      # --------------------
      # Diagnostics
      # --------------------
      # Memory Saving
      np.save('memories', self.memories)
      np.save('annealed_memories', self.annealed_memories)

      # Cost and Sensitivity Logging
      # logging.debug('deltaj: {}'.format(deltaj))
      # logging.debug('sensitivity: {}'.format(self.sensitivity))
      # monitor total cost
      # self.total_cost += np.sum(j1)
      # self.total_cost_steps += 1
      # logging.debug('total average cost: {}'.format(
      #   self.total_cost / self.total_cost_steps))

  def memorize(self, deltaj):
    '''
    Take the last segment of active_memory and certainty_memory
      and store it as a memory
    Structure of a memory:
      index 0: (np.array) active_memory -> used for pattern matching
      index 1: (np.array) certainty_memory -> used for adjusting certainty
      index 2: (int) deltaJ -> scaling certainty
      index 3: (boolean) is_annealed -> memories are not annealed by default
    '''
    memory = np.array([
      self.active_memory[-(2*self.cost_function_steps):],
      self.certainty_memory[-(2*self.cost_function_steps):] + self.base_certainty,
      deltaj,
      False
    ])
    if(self.memories is None):
      self.memories = np.array([memory])
    else:
      self.memories = np.vstack((self.memories, memory))
      
    # logging.debug('memory formed: {}'.format(memory))


  def decision_probability(self):
    '''
    Calculate decision probability, and call activate if passing threshold
    :return: Returns the certainty, as a float that can be negative
      (representing suppression) or positive (representing activation)
    :calls: Calls self.activate() if decides to activate
    '''
    # ---------------------
    # Sum Probabilities from memories
    # ---------------------
    activate_threshold = self.activate_threshold
    # Add in influence from memories
    activation_certainty = 0
    # record anenaled memories indecies to be cleared
    indecies_to_be_cleared = []
    # sum influences while popping them
    if(self.annealed_memories is not None):
      for (index, annealed) in np.ndenumerate(self.annealed_memories[:,0]):
        activation_certainty += annealed[0]
        self.annealed_memories[index[0]][0] = self.annealed_memories[index[0]][0][1:]
        # annealing is completed, remove and unmark annealed
        if(self.annealed_memories[index[0]][0].size == 0):
          # index[0] entry 1 stores the memory index[0], set this to not annealed
          self.memories[self.annealed_memories[index[0]][1]][3] = False

          indecies_to_be_cleared.append(index)
          # clear 
          if(self.annealed_memories.size == 0):
            self.annealed_memories = None

    # clear out any annealed memories that have terminated
    indecies_to_be_cleared.reverse()
    for i in indecies_to_be_cleared:
      self.annealed_memories = np.delete(self.annealed_memories, i, 0)
      if(self.annealed_memories.size == 0):
        self.annealed_memories = None

    activate_threshold += activation_certainty
    activated = False
    if(random() > activate_threshold):
      activated = True
      if(callable(self.activate)):
        self.activate(True)
    else:
      if(callable(self.activate)):
        self.activate(False)

    return activation_certainty
    # return activated

  def match_memories(self):
    '''
    Compare last n steps to first n steps of each memory and if they are similar
    enough, anneal them to the active memory
    '''
    if(self.memories is None):
      return
          
    highest_certainty = 0
    highest_certainty_index = None
    # iterate through the potential of each memory, use the highest certainty
    for (index, memory) in np.ndenumerate(self.memories[:, 0]):
      if(np.sum(np.power(memory[0:self.annealing_steps] - self.active_memory[-self.annealing_steps:],2)) < self.annealing_threshold):
        certainty = self.get_memory_certainty(index[0])
        # logging.debug('certainty: {}'.format(certainty))
        if(not self.is_annealed(index[0]) and certainty > highest_certainty):
          highest_certainty = certainty
          highest_certainty_index = index[0]

    if(highest_certainty_index is not None):
      # logging.debug('certainty of annealed memory: {}'.format(highest_certainty))
      self.anneal_memory(highest_certainty_index)

  def get_memory_certainty(self, index):
    '''
    Calculate the sum certainty of a memory

    '''
    memory = self.memories[index]
    return np.sum(np.abs(memory[1] * memory[2]))

  def is_annealed(self, index):
    '''
    Return if a memory is currently annealed (should not be multi-annealed)
    '''
    return self.memories[index][3]

  def anneal_memory(self, index):
    '''
    Anneal a memory to the active stream
    '''
    # logging.debug('annealing memory {}'.format(index))
    memory = self.memories[index]
    self.memories[index][3] = True
    annealing_memory = np.array(
      [memory[1][-self.cost_function_steps:] * memory[2], index])

    if(self.annealed_memories is None):
      # create certainty of this memory and directionality
      self.annealed_memories = np.array([annealing_memory])
    else:
      self.annealed_memories = np.vstack((self.annealed_memories, annealing_memory))

  def start(self):
    self.clock.start()
    
  def stop(self):
    logging.debug('neuron stopping')
    self.stopFlag.set()

  def receive_input(self, strength):
    self.input_potential += strength
  

  def init_potential_graph(self):
    '''
    Initialize matplotlib plot to graph potential on
    '''
    fig, ax = plt.subplots()
    ax.set_xlim([0, self.active_memory_length])
    ax.set_ylim([0, 1])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    x = np.linspace(0, self.active_memory_length, self.active_memory_steps)
    y = np.zeros(self.active_memory_steps)
    line, = plt.plot(x, y, animated=True)

    self.plot = {
      'fig': fig,
      'ax': ax,
      'line': line,
      'background': background
    }
    # plt.ioff()
    plt.show(block=False)


  def animate_potential(self):
    '''
    Animate own potential curve history in a thread
    '''
    self.plot['fig'].canvas.restore_region(self.plot['background'])
    if(len(self.active_memory) < self.active_memory_steps):
      y = np.concatenate([
        np.zeros(self.active_memory_steps - len(self.active_memory)),
        self.active_memory
      ])
    else: 
      y = self.active_memory

    self.plot['line'].set_ydata(y)
    self.plot['ax'].draw_artist(self.plot['line'])
    self.plot['fig'].canvas.blit(self.plot['ax'].bbox)
    