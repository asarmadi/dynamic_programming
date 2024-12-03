
import os
import copy
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DynamicProg:
    def __init__(self, sys_):
        self.system = sys_

        self.cost_to_go_current = np.zeros((self.system.grid_width, self.system.grid_height))
        self.cost_to_go_next = np.zeros((self.system.grid_width, self.system.grid_height))
        self.policy = np.zeros((self.system.grid_width, self.system.grid_height))
        self.plot_init()
        self.save_interval = 100
        self.out_dir = './out/'
        if not os.path.exists(self.out_dir+'Figs/'):
            os.makedirs(self.out_dir+'Figs/')
        if not os.path.exists(self.out_dir+'csv/'):
            os.makedirs(self.out_dir+'csv/')
        
    def discretize(self, state_, action_):
        state_ = np.array([self.system.q_dot_range[state_[0]],self.system.q_range[state_[1]]])
        next_state = self.system.step(state_,action_)
        if next_state[0] > 3 or next_state[1] > 3 or \
            next_state[0] < -3 or next_state[1] < -3:
            return None
        
        side = 'right'
        if np.random.random() <= 0.5:
            side = 'left'
        next_state[0] = np.searchsorted(self.system.q_dot_range, next_state[0], side=side)
        if side == 'right':
            next_state[0] -= 1

        side = 'right'
        if np.random.random() <= 0.5:
            side = 'left'
        next_state[1] = np.searchsorted(self.system.q_range, next_state[1], side=side)
        if side == 'right':
            next_state[1] -= 1

        return next_state.astype(int)

    def inside_goal(self, state_):
        if np.linalg.norm(state_-self.goal) < 1.5:
            return True
        return False

    def update_cost_to_go(self, state_):
        
        
        cost_to_go_l = []

        for action in self.system.actions:
            next_state = self.discretize(state_,action)
            if next_state is not None:
                cost_ = self.system.cost_function(state_, action)+self.cost_to_go_current[next_state[0],next_state[1]]
                
                cost_to_go_l.append(cost_)
        
        if cost_to_go_l != []:
            self.cost_to_go_next[state_[0],state_[1]] = np.min(cost_to_go_l)
        else:
            self.cost_to_go_next[state_[0],state_[1]] = 1 + np.max(self.cost_to_go_current)

    def plot_init(self):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='3d'))
        #self.ax.set_xticks(self.q_range)
        #self.ax.set_yticks(self.q_dot_range)
        #self.ax.set_xticklabels(self.q_range)
        #self.ax.set_yticklabels(self.q_dot_range)

    def plot_cost_to_go(self, iteration, save_fig):
        if not save_fig:
            self.plot_init()
        xs, ys = np.meshgrid(self.system.q_range, self.system.q_dot_range, sparse=True)
        #self.mesh = self.ax.pcolormesh(xs, ys, self.cost_to_go_current, linewidths=1,cmap='rainbow') # Plotting transpose to make sure the X, Y coordinates are shown correctly
        self.ax.plot_surface(xs, ys, self.cost_to_go_current, rstride=1, cstride=1, cmap=cm.jet)
        self.ax.set_title(f'Iteration: {iteration}')
        self.ax.set_xlabel(r'$q$')
        self.ax.set_ylabel(r'$\dot{q}$')

        if not save_fig:
            #cbar = self.fig.colorbar(self.mesh, ax=self.ax)
            plt.savefig(self.out_dir+'Figs/'+str(iteration)+'.png')
            np.savetxt(self.out_dir+'csv/'+str(iteration)+'.csv', self.cost_to_go_current, fmt="%.2f", delimiter=",")
            plt.show()
            plt.close()

    def plot_policy(self, iteration):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        xs, ys = np.meshgrid(self.system.q_range, self.system.q_dot_range, sparse=True)
        ax.plot_surface(xs, ys, self.policy, rstride=1, cstride=1, cmap=cm.jet)
        ax.set_title(f'Iteration: {iteration}')
        ax.set_xlabel(r'$q$')
        ax.set_ylabel(r'$\dot{q}$')

        plt.savefig(self.out_dir+'Figs/policy_'+str(iteration)+'.png')
        np.savetxt(self.out_dir+'csv/policy_'+str(iteration)+'.csv', self.policy, fmt="%.2f", delimiter=",")

        plt.show()
        plt.close()


    def update(self, frame, gen_animation):
        print(f"Update: {frame}")
        for i in range(self.system.grid_width):
            for j in range(self.system.grid_height):
                self.update_cost_to_go(np.array([i,j]))

        self.cost_to_go_current = copy.deepcopy(self.cost_to_go_next)

        for i in range(self.system.grid_width):
            for j in range(self.system.grid_height):
                action_dict = {}
                for idx, action in enumerate(self.system.actions):
                    next_state = self.discretize(np.array([i,j]),action)
                    if next_state is not None:
                        action_dict[idx] = self.cost_to_go_current[next_state[0],next_state[1]]
                if action_dict != {}:
                    min_key = min(action_dict, key=action_dict.get)
                    self.policy[i,j] = self.system.actions[min_key]

        if frame%self.save_interval == 0:
            self.plot_cost_to_go(frame,gen_animation)
            self.plot_policy(frame)
        if gen_animation:
            return self.mesh

    def run(self, n_iter, gen_animation):
        if gen_animation:
            ani = animation.FuncAnimation(self.fig, self.update, frames=n_iter, interval=500, repeat=False)
            ani.save(self.out_dir+'dp.gif',writer='pillow')
        else:
            for iter_idx in range(n_iter):
                self.update(iter_idx, gen_animation)
        return self.policy
        
        