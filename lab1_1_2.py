# author@litingyi


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

T = 15

WORLD_Y, WORLD_X = 5, 6
# left,up,right,down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25

reward = np.zeros((WORLD_Y, WORLD_X))
reward[2][3] = 1

def step(state, action):
    x_, y_ = state
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:# out of boundary
        next_state = state.tolist()
    if 0 <= x <= 2:
        if (y_== 1 and y == 2) or (y_ == 2 and y == 1):
            next_state = state.tolist()
        if 1 <= x <= 2:
            if (y_== 3 and y == 4) or (y_ == 4 and y == 3):
                next_state = state.tolist()
    if 4 <= y <= 5:
        if (x_== 1 and x == 2) or (x_ == 2 and x == 1):
            next_state = state.tolist()
    if 1 <= y <= 4:
        if (x_== 3 and x == 4) or (x_ == 4 and x == 3):
            next_state = state.tolist()
    if x == 4:
        if (y_ == 3 and y == 4) or (y_ == 4 and y == 3):
            next_state = state.tolist()

    return next_state


def probability(state, t):# return probability of minotaur in state at t step



def prob_suc(state, t, prob):# return probability of successfully arrive at state exactly at time t
    if (t == 0):
        if (state == (0, 0)): return prob
        else return 0
    else:
        # prob_suc(cur) = sum(prob_suc(prev)* prob(max q->a->cur)) * (1-probability(cur))
        prob_minotaur_in_state = probability(state, t) # fail
        action_returns = []
        for action in ACTIONS:
            (prev_x, prev_y) = step(state, action)
            if (next_x == x and next_y == y):
                pass# no appicable
            else:
                action_returns.append(action)

        dir_num = len(action_returns)# count all possible direction
        Dir_prob = 1/dir_num
        # for all prev:
        #    cur = prev
        prob_tmp = 0
        for action in action_returns:
            cur = state + action
            prob_tmp += Dir_prob * prob_suc(cur, t-1, prob) * prob(choose this direction)

        return prob_tmp * (1-prob_minotaur_in_state)


def value_iteration():
    # state value
    state_value = np.zeros((WORLD_Y, WORLD_X))# iniatil value function all to zero
    value = np.zeros(state_value.shape)
    # value iteration
    # Reward_calculate()
    print(state_value, "\n")
    print(reward, "\n")
    iteration = 0
    t = 0
    start = (4, 4)
    prev = (-1, -1) # no prev
    while true:
        for x in range(0, WORLD_Y):
            for y in range(0, WORLD_X):
                    action_returns = []
                    for action in ACTIONS:
                        (next_x, next_y) = step([x, y], action)
                        if (next_x == x and next_y == y):
                            action_returns.append(-float('inf'))
                        else:
                            if (next_x == 4 and next_y == 4):
                                action_returns.append(
                                    1 + 0.9 * state_value[next_x, next_y])
                            else:
                                action_returns.append(
                                    -1 + 0.9 * state_value[next_x, next_y])
                    new_value = np.max(action_returns)
                    state_value[x][y] = new_value
        if np.sum(np.abs(state_value - value)) < 1e-4:
            value = state_value.copy()
            break

        value = state_value.copy()
        iteration += 1

    print(iteration)


 # compute the optimal policy
    policy = np.zeros((WORLD_Y, WORLD_X))
    for x in range(0, WORLD_Y):
        for y in range(0, WORLD_X):
                action_returns = []
                for action in ACTIONS:
                    (next_x, next_y) = step([x, y], action)
                    if (next_x == x and next_y == y):
                        action_returns.append(-float('inf'))
                    else:
                        if (next_x == 2 and next_y == 3):
                            action_returns.append(
                                1 + 0.9 * state_value[next_x, next_y])
                        else:
                            action_returns.append(
                                -1 + 0.9 * state_value[next_x, next_y])
                policy[x, y] = np.argmax(action_returns)

    print(policy)
if __name__ == '__main__':
    value_iteration()
