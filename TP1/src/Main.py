import gym
import my_gym

import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from mazemdp.maze_plotter import show_videos
from my_gym.envs.maze_mdp import MazeMDPEnv
from mazemdp import random_policy

# For visualization
os.environ["VIDEO_FPS"] = "5"
if not os.path.isdir("./videos"):
    os.mkdir("./videos")

from IPython.display import Video

SIZE = 10
RATIO = 0.3

env = gym.make("MazeMDP-v0", kwargs={"width": SIZE, "height": SIZE, "ratio": RATIO})
env.reset()

# in dynamic programming, there is no agent moving in the environment
env.set_no_agent()
#env.init_draw("The maze")



def evaluate_v(mdp: MazeMDPEnv, policy: np.ndarray, v : np.array, k: int) -> np.ndarray:
    # Outputs the state value function of a policy
    stop = False
    v_updates = 0
    for i in range(k):
        vold = v.copy()
        v, v_step_updates = evaluate_one_step_v(mdp, vold, policy)
        v_updates += v_step_updates
        # Test if convergence has been reached
        if (np.linalg.norm(v - vold)) < 0.01:
            return v, v_updates
    return v, v_updates

def evaluate_one_step_v(mdp: MazeMDPEnv, v: np.ndarray, policy: np.ndarray) -> np.ndarray:
    # Outputs the state value function after one step of policy evaluation
    # Corresponds to one application of the Bellman Operator
    v_updates = 0
    v_new = v.copy()
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        if x not in mdp.terminal_states:
            # Process sum of the values of the neighbouring states
            summ = 0
            for y in range(mdp.nb_states):
                summ = summ + mdp.P[x, policy[x], y] * v[y]
            v_new[x] = mdp.r[x, policy[x]] + mdp.gamma * summ
            v_updates += 1
    return v_new, v_updates

def improve_policy_from_v(mdp: MazeMDPEnv, v: np.ndarray, policy: np.ndarray) -> np.ndarray:
    # Improves a policy given the state values
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        if x not in mdp.terminal_states:
            v_temp = np.zeros(mdp.action_space.size)
            for u in mdp.action_space.actions:
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    summ = summ + mdp.P[x, u, y] * v[y]
                v_temp[u] = mdp.r[x, u] + mdp.gamma * summ

            for u in mdp.action_space.actions:
                if v_temp[u] > v_temp[policy[x]]:
                    policy[x] = u
    return policy

# ---------------- Policy Iteration with the V function -----------------#
# Given an MDP, this algorithm simultaneously computes 
# the optimal state value function V and the optimal policy

def generalized_policy_iteration(mdp: MazeMDPEnv, k, render: bool = True) -> Tuple[np.ndarray, List[float]]:
    # policy iteration over the v function
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    v_list = []
    policy = random_policy(mdp)

    stop = False
    v_updates_total = 0
    iterations = 0

    if render:
        mdp.init_draw("Policy iteration V")

    while not stop:
        vold = v.copy()
        # Step 1 : Policy Evaluation
        v, v_updates = evaluate_v(mdp, policy, vold, k)
        v_updates_total += v_updates
        if render:
            mdp.draw_v_pi(v, policy, title="Policy iteration V")

        # Step 2 : Policy Improvement
        # À compléter...  
        policy = improve_policy_from_v(mdp, v, policy)
        
        # Check convergence
        if (np.linalg.norm(v - vold)) < 0.01:
            stop = True
        v_list.append(np.linalg.norm(v))
        iterations +=1

    if render:
        mdp.draw_v_pi(v, policy, title="Policy iteration V")
        mdp.mdp.plotter.video_writer.release()

    return v, v_list, v_updates_total, iterations


v, v_list, v_updates, iterations = generalized_policy_iteration(env, 10, render=True)