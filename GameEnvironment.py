import sys, pygame
import random
from Neuron import Neuron

class Player:
	# physics modes
	EASING = 'easing'
	ACCELERATION = 'acceleration'

	def __init__(self, screen):
		'''
		Initialize and draw player to screen
		:param screen_size: integer tuple
			Pass in screen_size
		'''
		# Player physics
		self.min_width = 40
		self.min_height = 40
		self.max_width = 100
		self.max_height = 100
		# relative inflate/deflate speeds as difference between max and min
		self.easing_inflate_speed = 0.15
		self.easing_deflate_speed = 0.15
		# alternate inflate/deflate speed changes calculation via acceleration
		self.inflate_acceleration = 0.3
		self.deflate_acceleration = 0.3
		# inflate mode (accelerate or easing i.e. relative)
		self.inflate_mode = self.EASING

		screen_size = screen.get_size()
		screen_width = screen_size[0]
		screen_height = screen_size[1]
		self.screen = screen

		# other basic information
		self.inflate_velocity = 0
		self.color = (0, 128, 255)
		self.x = (screen_width / 2) - (self.min_width / 2)
		self.y = (screen_height / 2) - (self.min_height / 2)
		# initialize rect at min_width and min_height
		self.rect = pygame.draw.rect(self.screen, self.color, 
			pygame.Rect(self.x, self.y, self.min_width, self.min_height))


	def round(self, value):
		'''
		round to nearest even number value
		'''
		return int(2 * round(float(value) / 2))

	def inflate(self, inflating):
		'''
		Inflate or shrink by single integer value
		:param inflating: boolean
			determines whether the player is inflating or deflating
		'''
		# relative inflation calculation
		if(self.inflate_mode == self.EASING):
			width = self.rect.width
			if(inflating):
				self.inflate_velocity = (self.max_width - width) * self.easing_inflate_speed
			else:
				self.inflate_velocity = (self.min_width - width) * self.easing_deflate_speed
			self.inflate_velocity = self.round(self.inflate_velocity)
			self.rect = self.rect.inflate(self.inflate_velocity, self.inflate_velocity)
			return self.rect

		elif(self.inflate_mode == self.ACCELERATION):
			width = self.rect.width
			if(inflating):
				self.inflate_velocity += self.inflate_acceleration
			else:
				self.inflate_velocity -= self.deflate_acceleration
				
			# if already max, inflate velocity is 0
			if((width == self.max_width and self.inflate_velocity >= 0)
				or (width == self.min_width and self.inflate_velocity <= 0)):
				self.inflate_velocity = 0

			inflate_change = self.inflate_velocity
			# if velocity will send past max_width, set to max, but keep velocity
			if(width + self.inflate_velocity <= self.min_width):
				inflate_change = self.min_width - width
			elif(width + self.inflate_velocity >= self.max_width):
				inflate_change = self.max_width - width

			inflate_change = self.round(inflate_change)
			self.rect = self.rect.inflate(inflate_change, inflate_change)
			return self.rect

	def draw(self):
		pygame.draw.rect(self.screen, self.color, self.rect)

	def proximity_sensor(self, enemies):
		'''
		Input sensor for neuron - calculate if any enemies are close to the edge
		Only detects the closest enemy
		'''
		closest_distance = 1000
		for enemy in enemies:
			distance = self.rect.left - enemy.rect.right
			if(distance < closest_distance):
				closest_distance = distance

		# ensure no negative or div by zero
		if(closest_distance < 1):
			closest_distance = 1
		if(closest_distance < 20):
			return (1 / closest_distance)
		else:
			return 0

	def move(self, direction):
		'''
		Move in the given direction with given speed as a tuple
		Not used in this game, maybe for another
		:return: resultant rectangle
		'''
		self.rect = self.rect.move(direction)
		return self.rect



class Enemy:
	'''
	Object actor to use in game. Stats defined by game environment
	'''
	def __init__(self, screen, rect = pygame.Rect(0, 115, 10, 10), 
		velocity = [2, 0]):
		self.rect = rect
		self.velocity = velocity
		self.screen = screen
		self.color = (255, 0, 0)

	def move(self):
		self.rect = self.rect.move(self.velocity)
		return self.rect

	def is_alive(self):
		screen_size = self.screen.get_size()
		screen_width = screen_size[0]
		screen_height = screen_size[1]
		if(self.rect.left < 0 or self.rect.right > screen_width
			or self.rect.top < 0 or self.rect.bottom > screen_height):
			return False
		else:
			return True

	def draw(self):
		pygame.draw.rect(self.screen, self.color, self.rect)


class Game:
	'''
	Game that creates the pygame screen and actors
	'''
	size = screen_width, screen_height = 320, 240
	black = 0, 0, 0
	clock = pygame.time.Clock()

	def __init__(self, environment_loop):
		'''
		:param environment_loop: function
			Function passed by GameEnvironment to run during loop
		'''
		pygame.init()

		self.screen = pygame.display.set_mode(self.size)
		self.player = Player(self.screen)
		self.enemies = []
		self.outputs = {
			'proximity': 0
		}
		self.environment_loop = environment_loop


	def read_controls(self):
		'''
		Read key presses and control player
		:return: True if escape pressed
		'''
		pressed = pygame.key.get_pressed()
		if(pressed[pygame.K_SPACE]):
			# inflate when space pressed
			self.player.inflate(True)
		else:
			# deflate when not pressed
			self.player.inflate(False)
		if(pressed[pygame.K_ESCAPE]):
			return True
		else:
			return False


	def spawn_enemy(self):
		'''
		Create a new enemy
		'''
		self.enemies.append(Enemy(self.screen))


	def redraw_game(self):
		'''
		Move actors and redraw the game
		'''
		self.screen.fill(self.black)
		self.player.draw()

		for (index, enemy) in enumerate(self.enemies):
			enemy.move()
			enemy.draw()
			if(enemy.rect.colliderect(self.player.rect)):
				enemy.velocity = [-self.player.inflate_velocity, 0]
			if(not enemy.is_alive()):
				self.enemies.pop(index)

		pygame.display.flip()

	def update_outputs(self):
		'''
		Update outputs for environment to access
		'''
		self.outputs['proximity'] = self.player.proximity_sensor(self.enemies)

	def maybe_spawn_enemy(self):
		spawn_chance = 0.01
		if(random.random() < spawn_chance):
			self.spawn_enemy()

	def start_loop(self):
		'''
		Start the game loop
		'''
		while True:
			for event in pygame.event.get():
				if(event.type == pygame.QUIT):
					return

			if(self.read_controls()):
				return
			self.redraw_game()
			self.maybe_spawn_enemy()

			self.clock.tick(100)
			self.update_outputs()
			self.environment_loop()	


class GameEnvironment:
	def __init__(self):
		self.game = Game(self.environment_loop)
		self.neuron = Neuron(timestep=0.02, active_memory_length=4.0, name="player",
			plot_potential=False)
		self.neuron.init_potential_graph()
		self.neuron.start()

	def environment_loop(self):
		self.neuron.receive_input(self.game.outputs['proximity'])
		self.neuron.animate_potential()

	def start(self):
		'''
		Starts the Game loop. When this loop ends, will exit
		'''
		self.game.start_loop()
		self.neuron.stop()
		sys.exit()

if(__name__ == '__main__'):
	game = GameEnvironment()
	game.start()