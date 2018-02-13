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
#    pass

#%%
class Neuron:
  # all of these parameters will have to deal with scaling in the future
  # may be best to do every one scaled in terms of timestep
  def __init__(self, timestep=1.0, active_memory_length=10.0, name='neuron',
      plot_potential=False, activate=None, activate_threshold=0.5,
      cost_function_window=1.0, delta_cost_threshold=5.0, base_certainty=0.001,
      annealing_steps=10, annealing_threshold=1.0):
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
    :param delta_cost_threshold: float optional (default = 5.0)
      The amount of cost change needed to trigger memorization
    :param base_certainty: float optional (default = 0.001)
      The base certainty applied to each memory, i.e. how quickly the neuron
      learns from experiences. Too high and it will fall into local optima
    :param annealing_steps: int, optional (default = 5)
      The number of time steps that memories should be compared to and annealed
      This may be better represented as a decimal of total steps in active
        memory
    :param annealing_threshold: float, optional (default = 1.0)
      Threshold of distance below which a memory should be considered a match to
      anneal
    '''
    self.memories = None
    self.active_memory = np.array([])
    self.certainty_memory = np.array([])
    self.active_memory_length = active_memory_length
    self.active_memory_steps = int(active_memory_length / timestep)
    self.input_potential = 0
    
    self.activate = activate
    self.activate_threshold = activate_threshold
    self.cost_function_window = cost_function_window
    self.cost_function_steps = int(cost_function_window / timestep)
    self.cost_steps_count = 0
    self.delta_cost_threshold = delta_cost_threshold
    self.base_certainty = base_certainty
    self.annealing_steps = annealing_steps
    self.annealing_threshold = annealing_threshold

    self.stopFlag = threading.Event()
    self.clock = Clock(timestep, self.stopFlag, self.step)
    self.clock.setName(name)

    self.plot_potential = plot_potential
    if(self.plot_potential):
      self.init_potential_graph()
    
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
    self.active_memory = np.append(self.active_memory, self.input_potential)
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
    '''
    cost_function = 'minimize'

    self.cost_steps_count += 1

    if(self.cost_steps_count >= self.cost_function_steps):
      self.cost_steps_count = 0
      steps = self.cost_function_steps
      j1 = self.active_memory[-steps:]
      j2 = self.active_memory[-(2 * steps):-steps]
      deltaj = np.sum(j2) - np.sum(j1)
      logging.debug('deltaj: {}'.format(deltaj))
      if(abs(deltaj) > self.delta_cost_threshold):
        self.memorize(deltaj)

  def memorize(self, deltaj):
    '''
    Take the last segment of active_memory and certainty_memory
      and store it as a memory
    Structure of a memory:
      index 0: (np.array) active_memory -> used for pattern matching
      index 1: (np.array) certainty_memory -> used for adjusting certainty
      index 2: (int) deltaJ -> scaling certainty
    '''
    memory = np.array([
      self.active_memory,
      self.certainty_memory + self.base_certainty,
      deltaj
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
    # activation_threshold += sum_of_memories
    # Calculate difference - this might be independent of base threshold?
    #   in which case self.activatae_threshold is not part of the eq.
    #   just alpha * sum_of_memories
    # activation_certainty = alpha * (self.activate_threshold - activate_threshold)
    activated = False
    if(random() > activate_threshold):
      activated = True
      if(callable(self.activate)):
        self.activate(True)
    else:
      if(callable(self.activate)):
        self.activate(False)

    # temp
    activation_certainty = 0

    return activation_certainty
    # return activated

  def match_memories(self):
    '''
    Compare last n steps to first n steps of each memory and if they are similar
    enough, anneal them to the active memory
    '''
    if(self.memories is None):
      return
      
    
    # # iterate through the potential of each memory
    for (index, memory) in np.ndenumerate(self.memories[:, 0]):
      if(np.sum(np.power(
        memory[0:self.annealing_steps] - self.active_memory[-self.annealing_steps:],
        2)) < self.annealing_threshold):
        print('ready to anneal memory #{}'.format(index))

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
    ax.set_ylim([0, 4])
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
    