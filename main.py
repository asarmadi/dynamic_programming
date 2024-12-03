import numpy as np
from utils.dp import DynamicProg

# We define an environment where the cost is 1 everywhere, except 0 at the goal and 10 at the obstacle
n_iterations = 101
system = 'DoubleIntegrator'

if system == 'DeathPit':
    from utils.deathPit import DeathPit
    grid_width, grid_height = 20, 20
    obstacles = np.array([[4,5],[7,7]])
    goal = np.array([8,2])
    sys_ = DeathPit(grid_width, grid_height, goal, obstacles)
elif system == 'DoubleIntegrator':
    from utils.doubleIntegrator import DoubleIntegrator
    sys_ = DoubleIntegrator()

#dp_obj = DynamicProg(sys_)
#policy_ = dp_obj.run(n_iterations, False)
policy_ = np.loadtxt('./out/csv/policy_100.csv', delimiter=",")

sys_.simulate(policy_, x0=np.array([26,23]))
