
# author@litingyi


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

T = 15

WORLD_Y, WORLD_X = 5, 6
# left,up,right,down,stay
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, 0])]

ACTIONS_MIN = [np.array([0, -1]),
               np.array([-1, 0]),
               np.array([0, 1]),
               np.array([1, 0])]

ACTION_PROB = 0.2


'''
# recursive reward
def Reward_calculate():
    # run into minotaur: reward -1
    state = np.array([4, 4])
    t = 0
    T = 3
    print(reward, "\n")

    def recursive(state, t, T):
        if t == T:
            reward[state[0], state[1]] += -10
        else:
            count = 4
            action_returns = []
            for action in ACTIONS:
                next = state + action
                next_state = next.tolist()
                x, y = next_state
                if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:
                    count -= 1
                else:
                    action_returns.append(action)

            if count != 0:
                # print(t, " ", T)
                for action in action_returns:
                    next = state + action
                    next_state = next.tolist()
                    x, y = next_state
                    reward[x, y] = np.round(-1/count * reward[state[0], state[1]], 3)
                    temp = t + 1
                    recursive(next_state, temp, T)
                    continue

    recursive(state, t, T)
    return 1
'''

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


def minotaur_step(state, action):
    x_, y_ = state
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:# out of boundary
        next_state = state.tolist()

    return next_state

def index(argument):

    if argument[0] == np.array([0, -1]):
        return 0
    if argument[0] == np.array([-1, 0]):
        return 1
    if argument[0] == np.array([0, 1]):
        return 2
    if argument[0] == np.array([1, 0]):
        return 3
    if argument[0] == np.array([0, 0]):
        return 4

    return -1

def value_iteration():
    # state value
    #reward = np.full((WORLD_Y, WORLD_X), 0)
    reward_win = 0.01 # win
    reward_death = -0.01 # death

    state_value = np.zeros((WORLD_Y, WORLD_X, WORLD_Y, WORLD_X))# me position & minotaur position
    policy = np.full((WORLD_Y, WORLD_X, WORLD_Y, WORLD_X), -1)
    value = np.zeros(state_value.shape)
    # value iteration
    #a = Reward_calculate()
    iteration = 0
    t = 0
    s1 = 0
    s2 = 0
    while t <= T:
        for m in range(0, WORLD_Y):
            for n in range(0, WORLD_X):
                # all possible positions of minotaur
                for x in range(0, WORLD_Y):
                    for y in range(0, WORLD_X):
                        # each position of me
                        if (m == x and n == y):
                            #or (m == next_x and n == next_y):
                        # minotaur not in the (x,y) and (next_x, next_y)
                            continue
                        action_returns = []
                        act_returns = []
                        for action in ACTIONS:
                            (next_x, next_y) = step([x, y], action)
                            if (next_x != x or next_y != y):
                                action_value = []
                                count = 4
                                # count of the num of min to go in each dir
                                for act in ACTIONS_MIN:
                                    (next_m, next_n) = minotaur_step([m, n], act)
                                    if (next_m == m and next_n == n):
                                        count -= 1 # this action is not applicable

                                prob_state = 1/count # prob of min going each dir

                                for act in ACTIONS_MIN:
                                    (next_m, next_n) = minotaur_step([m, n], act)
                                    if (next_m != m or next_n != n):
                                        if (next_m == next_x) and (next_n == next_y): # end up in the same cell
                                            action_value.append(np.round(prob_state * (reward_death + state_value[next_x, next_y]), 4))
                                        elif next_x == 4 and next_y == 4:# win
                                            action_value.append(np.round(prob_state * (reward_win + state_value[next_x, next_y]), 4))
                                        else:
                                            action_value.append(np.round(prob_state * (state_value[next_x, next_y]), 4))

                                action_returns.append(np.sum(action_value))
                                act_returns.append(action)
                                s1 = len(action_returns)
                                s2 = len(act_returns)

                        new_value = np.max(action_returns)
                        #policy[x][y][m][n] = index(act_returns[np.argmax(action_returns)])
                        state_value[x][y][m][n] = new_value

        #DELETE False later
        if False and np.sum(np.square(state_value - value)) < 1e-4:
            #value = state_value.copy()
            break
        else:
            value = state_value.copy()
            t += 1

    #print(t)


 # compute the optimal policy
    policy = []
    for m in range(0, WORLD_Y):
        for n in range(0, WORLD_X):
            # all possible positions of minotaur
            for x in range(0, WORLD_Y):
                for y in range(0, WORLD_X):
                    # each position of me
                    if (m == x and n == y):
                        #or (m == next_x and n == next_y):
                    # minotaur not in the (x,y) and (next_x, next_y)
                        continue

                    action_returns = []
                    act_returns = []
                    for action in ACTIONS:
                        (next_x, next_y) = step([x, y], action)
                        if (next_x != x or next_y != y):
                            action_value = []
                            count = 4
                            # count of the num of min to go in each dir
                            for act in ACTIONS_MIN:
                                (next_m, next_n) = minotaur_step([m, n], act)
                                if (next_m == m and next_n == n):
                                    count -= 1 # this action is not applicable

                            prob_state = 1/count # prob of min going each dir

                            for act in ACTIONS_MIN:
                                (next_m, next_n) = minotaur_step([m, n], act)
                                if (next_m != m or next_n != n):
                                    if (next_m == next_x) and (next_n == next_y): # end up in the same cell
                                        action_value.append(np.round(prob_state * (reward_death + state_value[next_x, next_y]), 4))
                                    elif next_x == 4 and next_y == 4:# win
                                        action_value.append(np.round(prob_state * (reward_win + state_value[next_x, next_y]), 4))
                                    else:
                                        action_value.append(np.round(prob_state * (state_value[next_x, next_y]), 4))

                            action_returns.append(np.sum(action_value))
                            act_returns.append(action)

            policy.append(act_returns[np.argmax(action_returns)])
            #policy(m, n, x, y)= policy[m*WORLD_Y+n*WORLD_X+x*WORLD_X + y]

    #For using at the simulation
    return policy

if __name__ == '__main__':
    value_iteration()