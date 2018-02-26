import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

memories = np.load('memories.npy')
annealed_memories = np.load('annealed_memories.npy')

# number_of_memories = memories.shape[0]
number_of_memories = 10
memory_start = 30
fig, ax = plt.subplots(number_of_memories, 2, sharex='col')

timesteps = memories[0][0].size

for i in range(memory_start, memory_start + number_of_memories):
	memory = memories[i]
	if(memory[2] > 0):
		color='green'
	elif(memory[2] < 0):
		color='red'

	# memory plot
	ax[i - memory_start][0].plot(range(timesteps), memory[0])
	ax[i - memory_start][0].get_yaxis().set_visible(False)
	# decision plot
	ax[i - memory_start][1].plot(range(timesteps), memory[1], c=color)
	ax[i - memory_start][1].get_yaxis().set_visible(False)

# for (i, item) in np.ndenumerate(memories):
# 	idx = i[0]
# 	if(memories[idx][2] > 0):
# 		color='green'
# 	elif(memories[idx][2] < 0):
# 		color='red'

# 	# memory plot
# 	if(i[1] == 0):
# 		ax[idx][0].plot(range(timesteps), item)
# 		# ax[idx][0].set_ylim(0, 1)
# 		ax[idx][0].get_yaxis().set_visible(False)
# 	# decision plot
# 	elif(i[1] == 1):
# 		ax[idx][1].plot(range(timesteps), item, c=color)
# 		# ax[idx][1].set_ylim(-1, 1)
# 		ax[idx][1].get_yaxis().set_visible(False)