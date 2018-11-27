#https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/cliff_walking.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 4

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# gamma for Q-Learning and Expected Sarsa
GAMMA = 0.8

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY]

# initial state action pair values
START = [0, 0]
BANK = [1, 1]
POLICE_START = [3, 3]

def move_police(police_position):
    moves = []
    position = police_position
    if position[0] > 0:
        moves.append(np.array([-1, 0]))

    if position[0] < WORLD_WIDTH -1:
        moves.append(np.array([1, 0]))

    if position[1] > 0:
        moves.append(np.array([0, -1]))

    if position[1] < WORLD_HEIGHT -1:
        moves.append(np.array([0, 1]))

    return position + random.choice(moves)

def step(state, action, police_position):
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    elif action == ACTION_STAY:
        next_state = state
    else:
        assert False

    police_position = move_police(police_position)

    reward = 0
    if next_state[0] == BANK[0] and next_state[1] == BANK[1]:
        reward = 1

    if next_state[0] == police_position[0] and next_state[1] == police_position[1]:
        reward = -10
        next_state = START
        police_position = POLICE_START

    return next_state, reward, police_position


# choose an action based on epsilon greedy algorithm
def choose_action(state,police_position, q_value, epsi=EPSILON):
    if np.random.binomial(1, epsi) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1],police_position[0],police_position[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

# an episode with Sarsa
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def sarsa(q_value, steps, episode, epsi = EPSILON):
    state = START
    police_position = POLICE_START

    action = choose_action(state,police_position, q_value, epsi)
    rewards = 0.0
    for i in range(steps):
        total_modif = steps*episode + i
        if total_modif == 0:
            step_size = ALPHA
        else:
            step_size = 1/(total_modif**(2/3))

        next_state, reward, next_police_position = step(state, action,police_position)
        next_action = choose_action(next_state,next_police_position, q_value, epsi)
        rewards += reward

        target = q_value[next_state[0], next_state[1],next_police_position[0],next_police_position[1], next_action]

        target *= GAMMA
        q_value[state[0], state[1],police_position[0],police_position[1], action] += step_size * (
                reward + target - q_value[state[0], state[1],police_position[0],police_position[1], action])
        state = next_state
        action = next_action
        police_position = next_police_position
    return rewards

# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, steps, episode):
    state = START
    police_position = POLICE_START
    rewards = 0.0

    for i in range(steps):
        total_modif = steps*episode + i
        if total_modif == 0:
            step_size = ALPHA
        else:
            step_size = 1/(total_modif**(2/3))

        action = choose_action(state,police_position, q_value)
        next_state, reward, next_police_position = step(state, action, police_position)
        rewards += reward
        # Q-Learning update
        q_value[state[0], state[1],police_position[0],police_position[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1],next_police_position[0],next_police_position[1], :]) -
                q_value[state[0], state[1],police_position[0],police_position[1], action])
        state = next_state
        police_position = next_police_position
    return rewards

# Use multiple runs instead of a single run and a sliding window
# With a single run I failed to present a smooth curve
# However the optimal policy converges well with a single run
# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def figure_6_4():
    # episodes
    episodes = 1000
    steps = 10000

    '''
    rewards_sarsa = []
    rewards_q_learning = []
    episod = []
    q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH,WORLD_HEIGHT, WORLD_WIDTH, 5))
    q_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH,WORLD_HEIGHT, WORLD_WIDTH, 5))

    for i in range(0, episodes):
        print("Episode: " + str(i))
        rewards_sarsa.append(sarsa(q_sarsa, steps, i))
        rewards_q_learning.append(q_learning(q_q_learning, steps, i))
        episod.append(i)

    # draw reward curves
    plt.plot(episod,rewards_sarsa, label='Sarsa')
    plt.plot(episod,rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()

    plt.savefig('./qlearning.png')
    plt.close()
    '''

    epsi = []
    mean_rew = []

    for j in range(1,20):
        print(j)
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH,WORLD_HEIGHT, WORLD_WIDTH, 5))
        rewards_sarsa = []
        epsilon = j/20
        for i in range(0, episodes):
            print("Episode: " + str(i))
            rewards_sarsa.append(sarsa(q_sarsa, steps, i, epsi))

        epsi.append(epsilon)
        mean_rew.append(sum(rewards_sarsa) / float(len(rewards_sarsa)))

    plt.plot(epsi,mean_rew)
    plt.xlabel('Epsilon')
    plt.ylabel('Mean reward')
    plt.legend()

    plt.savefig('./sarsa_epsilon.png')
    plt.close()


if __name__ == '__main__':
    figure_6_4()
