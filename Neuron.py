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
  def __init__(self, timestep=1.0, active_memory_length=10.0, name='neuron',
      plot_potential=False, activate=None, activate_threshold=0.5):
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
    '''
    self.memories = []
    self.active_memory = []
    self.active_memory_length = active_memory_length
    self.active_memory_steps = int(active_memory_length / timestep)
    self.input_potential = 0
    
    self.activate = activate
    self.activate_threshold = activate_threshold

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
    self.active_memory.append(self.input_potential)
    if(len(self.active_memory) > self.active_memory_steps):
      self.active_memory.pop(0)

    # --------------------
    # Match Memories
    # --------------------

    # --------------------
    # Calculate Decision
    # --------------------
    self.decision_probability()

    # --------------------
    # Calculate Cost Function
    # --------------------


    # logging.debug(self.active_memory)
    self.input_potential = 0

    if(self.plot_potential):
      self.animate_potential()
  
  def decision_probability(self):
    '''
    Calculate decision probability, and call activate if passing threshold
    :return: Returns the probability
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

    # return (activated, activation_certainty)
    return activated



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
    ax.set_ylim([0, 3])
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
    