#!/usr/bin/env python
# coding: utf-8

# In[64]:


def heuristic(node, goal):
    heuristic_values = {
        'A': 7,
        'B': 6,
        'C': 2,
        'D': 1,
        'E': 0
    }
    return heuristic_values[node]

def a_star_search(graph, start, goal):
    visited = set()
    priority_queue = PriorityQueue()
    priority_queue.put((0 + heuristic(start, goal), 0, start, [start]))

    while not priority_queue.empty():
        _, cost, node, path = priority_queue.get()

        if node == goal:
            print("Path:", "->".join(path))
            return

        if node in visited:
            continue

        visited.add(node)

        print("Visiting node:", node)
        print("Heuristic value:", heuristic(node, goal))

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_cost = cost + weight
                priority_queue.put((new_cost + heuristic(neighbor, goal), new_cost, neighbor, path + [neighbor]))

    print("Goal not reachable")

#example graph
graph = {
    'A': [('B', 4), ('C', 3)],
    'B': [('C', 1), ('D', 5)],
    'C': [('D', 2)],
    'D': [('E', 3)],
    'E': []
}

a_star_search(graph, 'A', 'E')


# In[ ]:





# In[ ]:




