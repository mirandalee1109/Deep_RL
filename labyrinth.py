
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib import collections  as mc
import random

SIZE_X=6
SIZE_Y=5

T=15

ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, 0])]

def min_move(position):

    moves = []
    if position[0] > 0:
        moves.append(np.array([-1, 0]))

    if position[0] < SIZE_X -1:
        moves.append(np.array([1, 0]))

    if position[1] > 0:
        moves.append(np.array([0, -1]))

    if position[1] < SIZE_Y -1:
        moves.append(np.array([0, 1]))

    return position + random.choice(moves)




def draw_image(player_path, min_path):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    ncols = SIZE_X
    nrows = SIZE_Y

    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for i in range(SIZE_Y):
        for j in range(SIZE_X):
            # Index either the first or second item of bkg_colors based on
            # a checker board pattern
            idx = [j % 2, (j + 1) % 2][i % 2]
            color = 'white'

            if i==4 and j==4:
                tb.add_cell(i, j, width, height,
                            loc='center', edgecolor='#63b1f2', facecolor='#9bb4db')

            elif i==0 and j==0:
                tb.add_cell(i, j, width, height,
                            loc='center', edgecolor='#63b1f2', facecolor='#9fe592')

            else:
                tb.add_cell(i, j, width, height,
                            loc='center', edgecolor='#63b1f2', facecolor=color)
    # Row Labels...
    for i, label in enumerate(range(SIZE_Y)):
        tb.add_cell(i, -1, width, height, text=label+1, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(range(SIZE_X)):
        tb.add_cell(-1, j, width, height/2, text=label+1, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)

    #Limits
    borders = [[(0, 0), (1, 0)], [(0, 0), (0, 1)], [(1, 1), (1, 0)], [(1, 1), (0, 1)]]
    lc = mc.LineCollection(borders, colors='k', linewidths=4)

    ax.add_collection(lc)

    walls = [[(2/6, 1), (2/6, 2/5)], [(4/6, 4/5), (4/6, 2/5)], [(4/6, 3/5), (1, 3/5)], [(1/6, 1/5), (5/6, 1/5)], [(4/6, 1/5), (4/6, 0)]]
    lc = mc.LineCollection(walls, colors='k', linewidths=2)

    ax.add_collection(lc)


    #Paths
    #May need improvements
    last = ((player_path[0][0] + 0.5)/6,(4.5-player_path[0][1])/5)
    path=[]
    for k in range(len(player_path)):
        path.append([last,((player_path[k][0] + 0.5)/6,(4.5-player_path[k][1])/5)])
        last = ((player_path[k][0] + 0.5)/6,(4.5-player_path[k][1])/5)

    lc = mc.LineCollection(path, colors='r', linewidths=2)

    ax.add_collection(lc)


    last = ((min_path[0][0] + 0.35)/6,(4.65-min_path[0][1])/5)
    path=[]
    for k in range(len(min_path)):
        path.append([last,((min_path[k][0] + 0.35)/6,(4.65-min_path[k][1])/5)])
        last = ((min_path[k][0] + 0.35)/6,(4.65-min_path[k][1])/5)

    lc = mc.LineCollection(path, colors='b', linewidths=2)

    ax.add_collection(lc)

    plt.savefig('./figure_4_1.png')
    plt.close()


def simulate(values, T):

    min_path=[[4,4]]
    player_path = [[0,0]]

    win = False

    for t in range(T):
        pos_min = min_path[-1]
        pos_player = player_path[-1]

        new_pos_player = pos_player + ACTIONS[values[pos_player[1]][pos_player[0]][pos_min[1]][pos_min[0]]]

        player_path.append(new_pos_player)
        min_path.append(min_move(pos_min))

        if new_pos_player[0] == 4 and new_pos_player[1] == 4:
            win = True
            break


    return player_path, min_path, win


if __name__ == '__main__':

    #REPLACE THIS WITH REAL VALUES
    values = np.zeros((SIZE_Y, SIZE_X, SIZE_Y, SIZE_X), dtype=int)
    values.fill(4)

    win_counter = 0

    #Example for drawing
    player_path, min_path, _ = simulate(values)

    total_simulations = 10000
    for i in range(total_simulations):
        _, _, win = simulate(values)

        if win:
            win_counter += 1

    draw_image(player_path, min_path)

    print("Total wins:" + str(win_counter))
    print("Out of:" + str(total_simulations))

    print("Done")
