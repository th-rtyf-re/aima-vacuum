# -*-coding:utf8-*-

import numpy as np
from heapq import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patches
from matplotlib.colors import from_levels_and_colors

# Remove MatPlotLib toolbar
matplotlib.rcParams['toolbar'] = 'None'

constants = {
	"air":			{"id": 0, "color": "#ffffff"},
	"block":		{"id": 1, "color": "#221022"},
	"dirt":			{"id": 2, "color": "#b18667"},
	"vacuum":		{"id": 3, "color": "#c00000"},
	"unknown":		{"id": 4, "color": "#a4dded"},
	"inaccessible":	{"id": 5, "color": "#014f63"},
	"DFS":			{"id": 6, "color": "#ffb5b6"},
	"A*":			{"id": 7, "color": "#ff5556"}
}


class UI():
	"""
	Helper class for displaying things onscreen. The user does not need to call
	this class directly.
	
	Note: there is a known issue that produces error messages if the display
	window is interactive, so we never allow that.
	(https://github.com/matplotlib/matplotlib/issues/21915)
	
	Attributes:
	- world_artist, ai_artist : MatPlotLib Artists.
	- vacuum_position_text : MatPlotLib Text. Gives a vacuum cleaner's position.
	"""
	
	def __init__(self, delay=1e-9, verbose=False):
		"""
		Initialize the UI.
		
		Parameters:
		- delay : None, or nonnegative float. Minimum time, in seconds, that
		any displayed matrix stays on screen.
		- verbose : bool. If `verbose`, then display log messages in the UI.
		"""
		self._delay = delay
		sorted_constants = sorted(constants.items(), key=lambda x: x[1]["id"])
		
		# Draw figure
		fig = plt.figure()
		patches = [Patch(color=v["color"], label=k) for k, v in sorted_constants]
		fig.legend(handles=patches, loc="center", bbox_to_anchor=(0.5, 0.1), ncol=4)
		
		# Draw two subplots
		ax = fig.subplots(ncols=2, sharex=True, sharey=True)
		ax[0].set_axis_off()
		ax[1].set_axis_off()
		ax[0].set_title("World view")
		ax[1].set_title("AI knowledge and search")
		
		# Create artists for the world view and the AI view
		levels = [v["id"] for k, v in sorted_constants]
		levels += [levels[-1] + 1]
		colors = [v["color"] for k, v in sorted_constants]
		cmap, norm = from_levels_and_colors(levels, colors)
		self.world_artist = ax[0].matshow([[0]], cmap=cmap, norm=norm, interpolation="none")
		self.ai_artist = ax[1].matshow([[0]], cmap=cmap, norm=norm, interpolation="none")
		
		# Create text elements
		text_table = ax[1].table(cellText=[" ", " "], loc="bottom", cellLoc="left", edges="open")
		self.vacuum_position_text = text_table[0, 0].get_text()
		self.log_text = type("dummy_log", (object,), dict(set_text=lambda x: None))
		if verbose:
			self.log_text = text_table[1, 0].get_text()
	
	def display(self, m, plot=0, overlay=None):
		"""
		Draw matrix `m` in a plot. If `overlay` is not `None`, then draw over
		the matrix with extra overlaid values.
		
		Parameters:
		- m : array-like. Matrix to be drawn by one of the object's artists.
		- plot : int. The choice of artist:
			* 0 : World artist,
			* 1 : AI artist.
		- overlay : None, or a pair (iterable of (int, int), int). Each
		element of the iterable is a position, the last int is the value to be
		overlaid. At each listed position, the artist will draw the overlaid
		value.
		
		Returns:
		- data : array-like, same type as `m`. The actual displayed matrix,
		with overlaid values.
		"""
		data = m.copy()
		if overlay is not None:
			for xy in overlay[0]:
				data[xy] = overlay[1]
		
		if plot == 0:
			self.world_artist.set_data(data)
		elif plot == 1:
			self.ai_artist.set_data(data)
		
		if self._delay:
			plt.pause(self._delay)
		return data
	
	def update_vacuum_position(self, pos):
		"""
		Update the displayed vacuum cleaner position.
		
		Parameters:
		- pos : int. The new position.
		"""
		self.vacuum_position_text.set_text("vacuum position: {}".format(pos))
	
	def log(self, message):
		"""
		Display `message` in the interface log.
		
		Parameters:
		- message : string. The log message.
		"""
		self.log_text.set_text(message)


class VacuumWorld():
	"""
	Class representing a world populated by vacuum cleaners. The world is
	encoded as a grid. Each square of the grid can be:
	- air : a navigable square for vacuum cleaners,
	- block : an obstructed square where vacuum cleaners cannot go,
	- dirt : a navigable square that can be cleaned by a vacuum cleaner.
	
	This class includes methods for generating elements in the world, as well
	as methods for controlling vacuum cleaners. Vacuum cleaners can only move
	in the four cardinal directions.
	
	Attributes:
	- grid : NumPy ndarray. A matrix with air, block, and dirt values. The
	actual values are given by the global dictionary `constants`. The border is
	always filled with blocks: this simplifies AI navigation and eliminates all
	edge cases.
	- vacuums : list of pairs of integers. A list of positions of vacuums.
	"""
	
	def __init__(self, shape, ui):
		"""
		Create a world of a given shape.
		
		Parameters:
		- shape : pair of integers. The size of the "real" world, without the
		border of blocks.
		- ui : UI object. The UI interface.
		"""
		self.grid = np.full((shape[0] + 2, shape[1] + 2), constants["block"]["id"], dtype=int)
		self._inner_grid = self.grid[1:-1, 1:-1]
		self.reset()
		
		self.vacuums = []
		self.ui = ui
	
	def get_true_shape(self):
		"""
		Returns:
		- A pair of int. The size of the actual world grid, border of blocks included.
		"""
		return self.grid.shape
	
	#### World creation methods
	
	def reset(self):
		"""
		Clear the world.
		"""
		self._inner_grid.fill(constants["air"]["id"])
	
	def gen_bernoulli(self, p, element):
		"""
		Generate elements in the world following a Bernoulli distribution. This
		overwrites any preexisting elements.
		
		Parameters:
		- p : float between 0 and 1. Probability parameter for the Bernoulli
		distribution.
		- element : string. Type of element to be generated.
		"""
		rng = np.random.default_rng()
		blocks = rng.binomial(1, p, size=self._inner_grid.shape)
		self._inner_grid[blocks == 1] = constants[element]["id"]
	
	def gen_constant(self, n, element):
		"""
		Generate a fixed number of elements in the world randomly. This
		overwrites any preexisting elements.
		
		Parameters:
		- n : nonnegative int. Number of elements to be generated.
		- element : string. Type of element to be generated.
		"""
		rng = np.random.default_rng()
		flat_idx = rng.choice(np.arange(self._inner_grid.size), size=n, replace=False)
		idx = np.unravel_index(flat_idx, self._inner_grid.shape)
		self._inner_grid[idx] = constants[element]["id"]
	
	def set_elements(self, positions, element):
		"""
		Place elements at given positions.
		
		Parameters:
		- positions : list of pairs of int. List of positions to place elements.
		- element : string. Type of element to be placed.
		"""
		for xy in positions:
			self.grid[xy] = constants[element]["id"]
	
	def add_vacuums(self, n):
		"""
		Add a specified number of vacuum cleaners, to be placed on distinct
		available squares.
		
		Parameters:
		- n : nonnegative int. Number of vacuum cleaners to be added.
		
		Returns:
		- A list of pairs of int. List of vacuum cleaner positions.
		"""
		rng = np.random.default_rng()
		available = np.nonzero(np.logical_or(self.grid == constants["air"]["id"], self.grid == constants["dirt"]["id"]))
		idx = rng.choice(np.arange(available[0].size), size=n, replace=False)
		self.vacuums = list(zip(available[0][idx], available[1][idx]))
		return self.vacuums
	
	#### Vacuum methods
	
	def vacuum_check(self, i):
		"""
		Check if the i^th vacuum cleaner is on a dirty square.
		
		Parameters:
		- i : int. Index of the vacuum cleaner.
		
		Returns:
		- A bool.
		"""
		return self.grid[self.vacuums[i]] == constants["dirt"]["id"]
	
	def vacuum_use(self, i):
		"""
		Use the i^th vacuum cleaner. We assume that the vacuum cleaner is in
		a valid position.
		
		Parameters:
		- i : int. Index of the vacuum cleaner.
		"""
		self.grid[self.vacuums[i]] = constants["air"]["id"]
	
	def vacuum_move(self, i, direction):
		"""
		Attempt to move the i^th vacuum cleaner in a specified direction.
		
		Parameters:
		- i : int. Index of the vacuum cleaner.
		- direction : string. The following four characters have meaning:
			* N : north, or up,
			* E : east, or left,
			* S : south, or down,
			* W : west, or right.
		
		Returns:
		- A bool. `True` if the vacuum successfully moved, `False` if not.
		"""
		x0, y0 = self.vacuums[i]
		x1, y1 = x0, y0
		if direction == 'N':
			x1 -= 1
		elif direction == 'E':
			y1 += 1
		elif direction == 'S':
			x1 += 1
		elif direction == 'W':
			y1 -= 1
		
		if self.grid[x1, y1] == constants["block"]["id"]:
			return False
		else:
			self.vacuums[i] = (x1, y1)
			self.ui.display(self.grid, plot=0, overlay=(self.vacuums, constants["vacuum"]["id"]))
			return True


class Vacuum():
	"""
	Class controlling a single vacuum cleaners. Users should interact with
	vacuum cleaners through this class.
	"""
	def __init__(self, world, i):
		"""
		Initialize the object by attaching it to the i^th vacuum cleaner of
		`world`.
		
		Parameters:
		- world: VacuumWorld object. The world of the vacuum cleaner.
		- i: int. The index of the vacuum cleaner in `world`.
		"""
		self._world = world
		self._i = i
	
	def check(self):
		"""
		Wrapper for `VacuumWorld.vacuum_check`.
		"""
		return self._world.vacuum_check(self._i)
	
	def use(self):
		"""
		Wrapper for `VacuumWorld.vacuum_use`.
		"""
		return self._world.vacuum_use(self._i)
	
	def move(self, direction):
		"""
		Wrapper for `VacuumWorld.vacuum_move`.
		
		Parameters:
		- direction : string. The following four characters have meaning:
			* N : north, or up,
			* E : east, or left,
			* S : south, or down,
			* W : west, or right.
		"""
		return self._world.vacuum_move(self._i, direction)


class SingleVacuumAI():
	"""
	Class that controls a vacuum cleaner in a world of air, blocks, and dirt.
	The vacuum cleaner belongs to a VacuumWorld, and we assume that it is the
	0th vacuum cleaner.
	
	Attributes:
	- world : VacuumWorld object. The World where the vacuum exists.
	`SingleVacuumAI` only has access to the vacuum cleaner methods of `world`.
	- space : NumPy ndarray. A grid of air, block, unknown, and inaccessible
	squares that represents the object's current knowledge.
	- start : pair of int. The starting position of the vacuum cleaner.
	- pos : pair of int. The current position of the vacuum cleaner.
	- ui : UI object. The UI interface.
	"""
	
	def __init__(self, vacuum, shape, vacuum_pos, ui):
		"""
		Initialize the AI with a vacuum cleaner's initial position.
		
		Parameters:
		- vacuum : Vacuum object. The vacuum cleaner to be controlled.
		- shape : pair of int. The shape of the world of the vacuum cleaner.
		- vacuum_pos : pair of int. The starting position of the vacuum cleaner.
		- ui : UI object. The UI interface.
		"""
		self.vacuum = vacuum
		self.space = np.full((shape[0] + 2, shape[1] + 2), constants["block"]["id"], dtype=int)
		self.space[1:-1, 1:-1] = constants["inaccessible"]["id"]
		
		self.start = vacuum_pos
		self.pos = self.start
		
		print("[AI] Starting vacuum AI at", self.start)
		self.space[self.pos] = constants["air"]["id"]
		self._update_accessible(self.pos)
		
		self.ui = ui
		self._space_overlaid = None
	
	def reset(self):
		"""
		Reset the knowledge space.
		"""
		self.ui.log("Reset knowledge space")
		self.space[1:-1, 1:-1] = constants["inaccessible"]["id"]
	
	def clean(self):
		"""
		Clean the entire world and return to the start.
		"""
		if self.vacuum.check():
			self.vacuum.use()
		self._dfs_exploration()
		print("[AI] Done cleaning, going home")
		self.ui.log("Searching for home with A*")
		self._go_to(self.start)
		self.ui.log("At home")
		self.ui.display(self.space, plot=1)
	
	def _dfs_exploration(self):
		"""
		Explore the grid with a greedy depth-first search method. For the
		problem of online graph exploration (we don't know anything about the
		graph beforehand), greedy algorithms are optimal: see "Online graph
		exploration: new results on old and new algorithms" by Megow, Mehlhorn,
		and Schweitzer (2012).
		
		We code imperatively, maintaining a stack of squares in the current
		branch being explored. While possible, we use an exploration heuristic
		(see `_exploration_heuristic`) to explore unknown squares. If all
		neighboring squares are known, then we backtrack to the most recent
		square with an unknown neighbor. Backtracking is done using an A*
		search algorithm (see `_astar`). This is marginally better than the
		simpler method of backtracking step by step to the oldest visited
		neighboring square.
		"""
		stack = [None]
		
		while stack:
			neighbors = self._unknown_neighbors(self.pos)
			if neighbors:
				unknown_square = max(neighbors, key=self._exploration_heuristic)
				moved = self.vacuum.move(self._cardinal_direction(self.pos, unknown_square))
				if moved:
					self.ui.log("moved, updating knowledge")
					if self.vacuum.check():
						self.vacuum.use()
					stack.append(self.pos)
					self.pos = unknown_square
					self.space[self.pos] = constants["air"]["id"]
					self._update_accessible(self.pos)
				else:
					self.ui.log("blocked, updating knowledge")
					self.space[unknown_square] = constants["block"]["id"]
			else:
				self.ui.log("backtracking with A*")
				prev_square = None
				while not neighbors:
					prev_square = stack.pop()
					if prev_square is None:
						return
					neighbors = self._unknown_neighbors(prev_square)
				self._go_to(prev_square)
			self.ui.update_vacuum_position(self.pos)
			self._space_overlaid = self.ui.display(self.space, plot=1, overlay=(stack[1:], constants["DFS"]["id"]))
	
	def _update_accessible(self, pos):
		"""
		Internal function updating inaccessible neighbors to unknown squares.
		
		Parameters:
		- pos : pair of int. Position of square.
		"""
		x, y = pos
		for neighbor in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
			if self.space[neighbor] == constants["inaccessible"]["id"]:
				self.space[neighbor] = constants["unknown"]["id"]
	
	def _unknown_neighbors(self, pos):
		"""
		Internal function getting list of unknown neighbors.
		
		Parameters:
		- pos : pair of int. Position of square.
		
		Returns:
		- A list of pairs of int. List of unknown neighbors.
		"""
		x, y = pos
		neighbors = []
		for xy in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
			if self.space[xy] == constants["unknown"]["id"]:
				neighbors.append(xy)
		return neighbors
	
	def _cardinal_direction(self, from_pos, to_pos):
		"""
		Internal function computing cardinal direction from `from_pos` to
		`to_pos`. The cardinal directions are
			* N : north, or up,
			* E : east, or left,
			* S : south, or down,
			* W : west, or right.
		
		Parameters:
		- from_pos, to_pos : pairs of int. Positions of squares in question.
		
		Returns:
		- a string of length 1. Cardinal direction.
		"""
		x0, y0 = from_pos
		x1, y1 = to_pos
		return " SEWN"[x1 - x0 + 2 * (y1 - y0)]  # very hacky
	
	def _exploration_heuristic(self, candidate_pos):
		"""
		Internal function computing a heuristic when exploring a branch in
		depth-first search. We use the following heuristics:
		- Favor squares with many known neighbors,
		- Favor increasing distance from the start.
		
		Parameters:
		- candidate_pos : pair of int. Coordinates of a candidate unknown
		square.
		
		Return :
		- A pair of int. Number of known neighbors, and 1-norm distance from
		start.
		"""
		x, y = candidate_pos
		n_known = 0
		for xy in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
			if self.space[xy] == constants["air"]["id"] or self.space[xy] == constants["block"]["id"]:
				n_known += 1
		
		return (n_known, abs(x - self.start[0]) + abs(y - self.start[1]))
	
	def _go_to(self, to_pos):
		"""
		Follow a path in known territory. Uses an A* search algorithm (see
		`_astar`).
		
		Parameters:
		- to_pos : pair of int. Target square.
		"""
		path = self._astar(self.pos, to_pos)
		for i in range(len(path) - 1):
			self.vacuum.move(self._cardinal_direction(path[i], path[i + 1]))
		self.pos = to_pos
		self.ui.update_vacuum_position(self.pos)
	
	def _astar(self, from_pos, to_pos):
		"""
		A* algorithm from from_pos to to_pos using 1-norm distance as the
		heuristic. We only travel on known squares. Since the heuristic is
		admissible, we are guaranteed to get the optimal solution.
		
		Parameters:
		- from_pos, to_pos : pairs of int. Start and end squares.
		
		Output:
		- A list of pairs of int. A path from from_pos to to_pos of shortest
		length.
		"""
		prev_square = np.full(self.space.shape + (2,), -1)
		prev_square[from_pos] = from_pos
		counter = 0
		candidates = []
		best_candidate = (0, counter, 0, from_pos)
		
		while best_candidate[-1] != to_pos:
			x, y = best_candidate[-1]
			neighbors = []
			for xy in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
				if self.space[xy] == constants["air"]["id"] and prev_square[xy][0] == -1:
					neighbors.append(xy)
					prev_square[xy] = (x, y)
			for xy in neighbors:
				backward_cost = best_candidate[2] + 1
				cost = backward_cost + abs(xy[0] - to_pos[0]) + abs(xy[1] - to_pos[1])
				counter += 1
				heappush(candidates, (cost, counter, backward_cost, xy))
				self.ui.display(self._space_overlaid, plot=1, overlay=((c[-1] for c in candidates), constants["A*"]["id"]))
			if candidates:
				best_candidate = heappop(candidates)
			else:
				print("[AI] A* did not find a path, producing an artificial path")
				return [from_pos]
		
		pos = best_candidate[-1]
		best_path = [pos]
		while pos != from_pos:
			pos = tuple(prev_square[pos])
			best_path.append(pos)
		self.ui.display(self._space_overlaid, plot=1, overlay=(best_path, constants["A*"]["id"]))
		return list(reversed(best_path))


def random_simulation(shape, density, dirtiness, delay=1e-9, verbose=True):
	"""
	Generate a randomized world and run the vacuum cleaner AI.
	
	Parameters:
	- shape : pair of int. Shape of the world.
	- density : float between 0 and 1. Density of blocks.
	- dirtiness : float between 0 and 1. Density of dirt.
	"""
	ui = UI(delay, verbose)
	
	world = VacuumWorld(shape, ui)
	world.gen_bernoulli(dirtiness, "dirt")
	world.gen_bernoulli(density, "block")
	vacuum_pos = world.add_vacuums(1)[0]
	
	vacuum = Vacuum(world, 0)
	ai = SingleVacuumAI(vacuum, shape, vacuum_pos, ui)
	ai.clean()
	print("Enter anything to close the UI: ", end="")
	input()

if __name__ == "__main__":
	#random_simulation((10, 10), .3, .5, delay=.5)
	random_simulation((70, 70), .35, .5)