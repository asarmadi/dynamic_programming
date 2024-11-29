
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DeathPit:
    def __init__(self, grid_width, grid_height, goal, obstacles):
        self.grid_width, self.grid_height = grid_width, grid_height
        self.goal = goal
        self.obstacles = obstacles
        self.cost_to_go_current = np.zeros((grid_width, grid_height))
        self.cost_to_go_next = np.zeros((grid_width, grid_height))
        self.policy = np.zeros((grid_width, grid_height,2))
        self.actions = np.array([[-1,0],[1,0],[0,-1],[0,1]]) # Up, Down, Left, Right
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(1,self.grid_height+1))
        self.ax.set_yticks(np.arange(1,self.grid_width+1))
        self.ax.set_xticklabels(np.arange(1,self.grid_height+1))
        self.ax.set_yticklabels(np.arange(1,self.grid_width+1))
        self.save_interval = 1
        self.out_dir = './out/'
        if not os.path.exists(self.out_dir+'Figs/'):
            os.makedirs(self.out_dir+'Figs/')
        if not os.path.exists(self.out_dir+'csv/'):
            os.makedirs(self.out_dir+'csv/')

    def cost_function(self, state_, action_):
        if self.inside_goal(state_): ## Checking for the goal
            return 0
        elif self.inside_obstacle(state_): ## This is the location of the pit
            return 10
        else:
            return 1
        
    def dynamics(self, state_, action_):
        next_state = state_ + action_
        if next_state[0] >= self.grid_width or next_state[1] >= self.grid_height or \
            next_state[0] < 0 or next_state[1] < 0:
            return None
        return next_state

    def inside_obstacle(self, state_):
        if state_[0] >= self.obstacles[0][0] and state_[0] <= self.obstacles[1][0] and \
            state_[1] >= self.obstacles[0][1] and state_[1] <= self.obstacles[1][1]:
            return True
        return False

    def inside_goal(self, state_):
        if (state_ == self.goal).all():
            return True
        return False

    def update_cost_to_go(self, state_):
        if self.inside_goal(state_):
            self.cost_to_go_next[state_[0],state_[1]] = 0
            return
        
        cost_to_go_l = []

        for action in self.actions:
            next_state = self.dynamics(state_,action)
            if next_state is not None:
                cost_ = self.cost_function(state_, action)+self.cost_to_go_current[next_state[0],next_state[1]]
                cost_to_go_l.append(cost_)
        
        self.cost_to_go_next[state_[0],state_[1]] = np.min(cost_to_go_l)

    def plot_cost_to_go(self, iteration, save_fig):
        if not save_fig:
            self.fig, self.ax = plt.subplots()
        
        x, y = np.arange(1,self.grid_height+1), np.arange(1,self.grid_width+1) # Changing the zero-index to 1
        self.mesh = self.ax.pcolormesh(x, y , self.cost_to_go_current, linewidths=1,cmap='rainbow') # Plotting transpose to make sure the X, Y coordinates are shown correctly
        X, Y = np.meshgrid(x, y)
        U = self.policy[:,:,1]
        V = self.policy[:,:,0]
        self.quiver = self.ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
        self.ax.set_title(f'Iteration: {iteration}')
        if not save_fig:
            cbar = self.fig.colorbar(self.mesh, ax=self.ax)
            plt.savefig(self.out_dir+'Figs/'+str(iteration)+'.png')
            np.savetxt(self.out_dir+'csv/'+str(iteration)+'.csv', self.cost_to_go_current, fmt="%.2f", delimiter=",")
            plt.close()

    def update(self, frame, gen_animation):
        print(f"Update: {frame}")
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                self.update_cost_to_go(np.array([i,j]))
        self.cost_to_go_current = self.cost_to_go_next
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                action_dict = {}
                for idx, action in enumerate(self.actions):
                    next_state = self.dynamics(np.array([i,j]),action)
                    if next_state is not None:
                        action_dict[idx] = self.cost_to_go_current[next_state[0],next_state[1]]
                min_key = min(action_dict, key=action_dict.get)
                self.policy[i,j,:] = self.actions[min_key]

        if frame%self.save_interval == 0:
            self.plot_cost_to_go(frame,gen_animation)
        if gen_animation:
            return self.mesh, self.quiver

    def run(self, n_iter, gen_animation):
        if gen_animation:
            ani = animation.FuncAnimation(self.fig, self.update, frames=n_iter, interval=500, repeat=False)
            ani.save(self.out_dir+'dp.gif',writer='pillow')
        else:
            for iter_idx in range(n_iter):
                self.update(iter_idx, gen_animation)
        
        