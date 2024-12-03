import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DoubleIntegrator:
    def __init__(self):
        self.dt = 0.01
        self.grid_height, self.grid_width = 32, 52
        self.goal = np.array([self.grid_width//2,self.grid_height//2])
        self.q_max, self.q_dot_max = 3, 3
        self.q_range     = np.linspace(-self.q_max,self.q_max,self.grid_height)
        self.q_dot_range = np.linspace(-self.q_dot_max,self.q_dot_max,self.grid_width)
        self.actions = np.linspace(-1,1,9)
        self.state_traj = []


    def cost_function(self, state_, action_):
        if np.linalg.norm(state_-self.goal) < 1.5: ## Checking for the goal
            return 0
        else:
            return 1
        
    def step(self, state_, action_):
        next_state = state_ + self.dt*np.array([action_,state_[0]])
        return next_state

    
    
    def discretize_state(self, state_):
        if state_[0] > self.q_dot_max or state_[1] > self.q_max or \
            state_[0] < -self.q_dot_max or state_[1] < -self.q_max:
            return None
        
        side = 'right'
        if np.random.random() <= 0.5:
            side = 'left'
        state_[0] = np.searchsorted(self.q_dot_range, state_[0], side=side)
        if side == 'right':
            state_[0] -= 1

        side = 'right'
        if np.random.random() <= 0.5:
            side = 'left'
        state_[1] = np.searchsorted(self.q_range, state_[1], side=side)
        if side == 'right':
            state_[1] -= 1

        return state_.astype(int)

    def update_animation(self, frame):
        x     = self.state_traj[frame]
        self.cart.set_data([x - 0.5, x + 0.5], [0, 0])
        return self.cart

    def simulate(self, policy_, x0):
        self.state_traj = []
        for i in range(20):
            print(x0)
            u = policy_[x0[0],x0[1]]
            x0 = np.array([self.q_dot_range[x0[0]],self.q_range[x0[1]]])
            x_ = self.step(x0, u)
            self.state_traj.append(x_[1])
            x0 = self.discretize_state(x_)

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid()
        
        # Create cart and pole objects
        self.cart, = self.ax.plot([], [], 'o-', lw=2)
        ani = animation.FuncAnimation(self.fig, self.update_animation, frames=len(self.state_traj), interval=self.dt*1000, repeat=False)

        gif_writer = animation.ImageMagickWriter(fps=20)
        # Uncomment the line below to save the animation
        ani.save('./out/double_integrator_animation.gif',writer='pillow')
        
        plt.show()
