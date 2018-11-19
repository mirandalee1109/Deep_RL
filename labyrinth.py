
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib import collections  as mc

rewards = {}

SIZE_X=6
SIZE_Y=5

#Move:
#1:up
#2:right
#3:down
#4:left
def valid_movement(position,move):

    if move == 1:
        if position[1] == 1:
            return False
        elif position[1] == 5 and (position[0] == 2 or position[0] == 3 or position[0] == 4 or position[0] == 5):
            return False
        elif position[1] == 3 and (position[0] == 5 or position[0] == 6):
            return False
        else:
            return True
    elif move == 2:
        if position[0] == 6:
            return False
        elif position[0] == 2 and (position[1] == 1 or position[1] == 2 or position[1] == 3):
            return False
        elif position[0] == 4 and (position[1] == 2 or position[1] == 3 or position[1] == 5):
            return False
        else:
            return True
    elif move == 3:
        if position[1] == 5:
            return False
        elif position[1] == 4 and (position[0] == 2 or position[0] == 3 or position[0] == 4 or position[0] == 5):
            return False
        elif position[1] == 2 and (position[0] == 5 or position[0] == 6):
            return False
        else:
            return True
    else:
        if position[0] == 1:
            return False
        elif position[0] == 3 and (position[1] == 1 or position[1] == 2 or position[1] == 3):
            return False
        elif position[0] == 5 and (position[1] == 2 or position[1] == 3 or position[1] == 5):
            return False
        else:
            return True


def average_reward(position):
    try:
        value = rewards[str(position)]
    except KeyError:
        return "Not found, calculating"
    else:
        return value


def draw_image(image, player_path, min_path):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(image):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = 'white'

        tb.add_cell(i, j, width, height, text=val,
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
    last = ((player_path[0][0] - 0.5)/6,(5.5-player_path[0][1])/5)
    path=[]
    for k in range(len(player_path)):
        path.append([last,((player_path[k][0] - 0.5)/6,(5.5-player_path[k][1])/5)])
        last = ((player_path[k][0] - 0.5)/6,(5.5-player_path[k][1])/5)

    lc = mc.LineCollection(path, colors='r', linewidths=2)

    ax.add_collection(lc)


    last = ((min_path[0][0] - 0.5)/6,(5.5-min_path[0][1])/5)
    path=[]
    for k in range(len(min_path)):
        path.append([last,((min_path[k][0] - 0.5)/6,(5.5-min_path[k][1])/5)])
        last = ((min_path[k][0] - 0.5)/6,(5.5-min_path[k][1])/5)

    lc = mc.LineCollection(path, colors='b', linewidths=2)

    ax.add_collection(lc)

if __name__ == '__main__':

    position=[0,0,0,0]
    #rewards[str(position)]=1

    #print(average_reward(position))

    values = np.zeros((SIZE_Y, SIZE_X))

    player_path = [[1,1],[1,2],[1,3],[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],[6,5],[5,5]]
    #min_path = [[5,5],[5,4],[4,4],[4,3],[5,3],[5,2],[4,2],[4,3],[3,3],[2,3],[2,2]]

    min_path=[[5,5]]
    draw_image(values, player_path, min_path)

    plt.savefig('./figure_4_1.png')
    plt.close()

    print("Done")
