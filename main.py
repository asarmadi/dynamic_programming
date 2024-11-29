import numpy as np
from utils.system import DeathPit
from tqdm import tqdm
# We define an environment where the cost is 1 everywhere, except 0 at the goal and 10 at the obstacle
n_iterations = 30
grid_width, grid_height = 20, 20
obstacles = np.array([[4,5],[7,7]])
goal = np.array([8,2])
sys_ = DeathPit(grid_width, grid_height, goal, obstacles)
sys_.run(n_iterations, False)
