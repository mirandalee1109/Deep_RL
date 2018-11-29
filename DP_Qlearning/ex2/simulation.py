
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib import collections  as mc
import random
import os

#from calculation import value_iteration
import test
from test import value_iteration_inf

WORLD_X=6
WORLD_Y=3
bank_pos = ((0, 0), (5, 0), (0, 2), (5, 2))
# left,up,right,down,stay
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, 0])]

ACTIONS_police = [np.array([0, -1]),
                  np.array([-1, 0]),
                  np.array([0, 1]),
                  np.array([1, 0])]


# left,up,right,down,stay
def policy_dir(robber, police):
    x, y = robber
    m, n = police
    action = []
    if x < m:
        if y < n:
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action
        elif y > n:
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action
        else:#y==n
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m > 0:
                action.append(ACTIONS_police[1])#up
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            return action
    elif x > m:
        if y < n:
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            return action
        elif y > n:
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            return action
        else:
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            return action
    else:#x == m
        if y < n:
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action
        else:
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action



#Moving the police toward robber
def police_move(position, actions):
    pos = np.array(position)
    act = random.choice(actions)
    next_state = (pos + act).tolist()
    return next_state

def robber_step(position, actions):
    pos = np.array(position)
    next_state = (pos + actions).tolist()
    '''
    if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:
        # out of boundary
        next_state = pos.tolist()
        flag = False
    '''
    return next_state

#Simulating with inf time and action grid given
def simulate_inf(policy):
    #initialzed positions
    police_path=[[2,1]]
    robber_path = [[0,0]]

    #Checking if we have won or not
    caught = False
    reward = 0

    while True:
        print("---------------")
        #Where each one is
        police_pos = police_path[-1]
        robber_pos = robber_path[-1]
        #print(police_pos, " ", robber_pos)
        a = policy[robber_pos[0]][robber_pos[1]][police_pos[0]][police_pos[1]]
        #Moving the player
        new_robber_pos = robber_step(robber_pos, ACTIONS[a])
        print(a)
        print(new_robber_pos)
        actions = policy_dir(robber_pos, police_pos)
        #print(actions)
        new_police_pos = police_move(police_pos, actions)
        print(new_police_pos)
        robber_path.append(new_robber_pos)
        police_path.append(new_police_pos)
        #print(reward)

        #If caught
        if new_police_pos[0] == new_robber_pos[0] and new_police_pos[1] == new_robber_pos[1]:
            caught = True
            reward -= 50
            break
        elif (new_robber_pos[0], new_robber_pos[1]) in bank_pos:
            reward += 10


    return robber_path, police_path, reward


if __name__ == '__main__':


    #Example for drawing
    policy = value_iteration_inf()
    #
    #
    robber_path, police_path, reward = simulate_inf(policy)
    #print(robber_path)
    #print(police_path)
    #print(reward)
    #
    #
    time = []
    wins = []



'''
    #Simulations
    for t in range(0,120):
        win_counter = 0
        total_simulations = 10000
        policy = value_iteration(t)

        for i in range(total_simulations):
            _, _, win = simulate(policy, t)

            if win:
                win_counter += 1

        print("T:" + str(t))
        print("Total wins:" + str(win_counter))
        print("Out of:" + str(total_simulations))
        print("-------------------------")
        time.append(t)
        wins.append(float(win_counter)/(total_simulations/100))

    plt.plot(np.asarray(time),np.asarray(wins))
    plt.xlabel("T")
    plt.ylabel("Win %")
    plt.savefig('./graph.png')
    plt.close()

    print("Done")
'''
