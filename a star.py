#!/usr/bin/env python
# coding: utf-8

# In[63]:


import heapq

def heuristic(node, goal):
    # Define your heuristic function here
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar(graph, start, goal):
    frontier = [(0, start)]
    heapq.heapify(frontier)
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next_node in graph[current]:
            new_cost = cost_so_far[current] + graph[current][next_node]
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()

    return path

# Example usage
graph = {
    (0, 0): {(0, 1): 1, (1, 0): 1},
    (0, 1): {(0, 0): 1, (1, 1): 1},
    (1, 0): {(0, 0): 1, (1, 1): 1},
    (1, 1): {(0, 1): 1, (1, 0): 1}
}

start = (0, 0)
goal = (1, 1)

print(astar(graph, start, goal))


# In[ ]:





# In[ ]:




