from enum import Enum
from queue import PriorityQueue
import numpy as np
from math import sqrt


def createGrid(data, droneAltitude, safetyDistance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    northMin = np.floor(np.min(data[:, 0] - data[:, 3]))
    northMax = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    eastMin = np.floor(np.min(data[:, 1] - data[:, 4]))
    eastMax = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    northSize = int(np.ceil(northMax - northMin))
    eastSize = int(np.ceil(eastMax - eastMin))

    # Initialize an empty grid
    grid = np.zeros((northSize, eastSize))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, dNorth, dEast, dAlt = data[i, :]
        if alt + dAlt + safetyDistance > droneAltitude:
            obstacle = [
                int(np.clip(north - dNorth - safetyDistance - northMin, 0, northSize-1)),
                int(np.clip(north + dNorth + safetyDistance - northMin, 0, northSize-1)),
                int(np.clip(east - dEast - safetyDistance - eastMin, 0, eastSize-1)),
                int(np.clip(east + dEast + safetyDistance - eastMin, 0, eastSize-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(northMin), int(eastMin)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    SOUTHWEST = (1, 1, sqrt(2))
    NORTHWEST = (-1, -1, sqrt(2))
    EAST = (0, 1, 1)
    SOUTHEAST = (1, -1, sqrt(2))
    NORTHEAST = (-1, 1, sqrt(2))
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def validActions(grid, currentNode):
    """
    Returns a list of valid actions given a grid and current node.
    """
    validActions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = currentNode

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        validActions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        validActions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        validActions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        validActions.remove(Action.EAST)
    if x+1 > n or y-1 < 0 or grid[x+1, y-1] == 1:
        validActions.remove(Action.SOUTHWEST)
    if x-1 < 0 or y-1 < 0 or grid[x-1, y-1] == 1:
        validActions.remove(Action.NORTHWEST)
    if x+1 > n or y+1 > m or grid[x+1, y+1] == 1:
        validActions.remove(Action.SOUTHEAST)
    if x-1 < 0 or y+1 > m or grid[x-1, y+1] == 1:
        validActions.remove(Action.NORTHEAST)


    return validActions


def aStar(grid, h, start, goal):
    
    path = []
    pathCost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        currentNode = item[1]
        
        if currentNode == start:
            currentCost = 0.0
        else:              
            currentCost = branch[currentNode][0]
             
        if currentNode == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in validActions(grid, currentNode):
                # get the tuple representation
                da = action.delta
                nextNode = (currentNode[0] + da[0], currentNode[1] + da[1])
                branchCost = currentCost + action.cost
                queueCost = branchCost + h(nextNode, goal)
                
                if nextNode not in visited:                
                    visited.add(nextNode)               
                    branch[nextNode] = (branchCost, currentNode, action)
                    queue.put((queueCost, nextNode))
                    
          
    if found:
        # retrace steps
        n = goal
        pathCost = branch[n][0]
        path.append(goal)
        
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], pathCost



def heuristic(position, goalPosition):
    h = np.linalg.norm(np.array(position) - np.array(goalPosition))
    #print("I arrived here: {}".format(h))
    return h

