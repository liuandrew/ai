# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:49:47 2018

@author: Andy
"""

import logging
import threading
import numpy as np
import matplotlib

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
      plot_potential=False):
    '''
    Construct a new Neuron
    :param timestep: float, optional (default=1.0)
      How often the neuron processes it's time in seconds.
      this defines how often it can fire as well
    :param active_memory_length: float, optional (default=10.0)
      The length of time in seconds the neuron holds in active memory
    :param name: string, optional (default='neuron')
    '''
    self.memories = []
    self.active_memory = []
    self.active_memory_length = active_memory_length
    self.active_memory_steps = int(active_memory_length / timestep)
    self.input_count = 0
    
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
    # logging.debug('input_count: {}'.format(self.input_count))
    self.active_memory.append(self.input_count)
    if(len(self.active_memory) > self.active_memory_steps):
      self.active_memory.pop(0)

    # logging.debug(self.active_memory)
    self.input_count = 0

    if(self.plot_potential):
      self.animate_potential()
  
  def start(self):
    self.clock.start()
    
  def stop(self):
    logging.debug('neuron stopping')
    self.stopFlag.set()

  def receive_input(self, strength):
    self.input_count += strength
  

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
    