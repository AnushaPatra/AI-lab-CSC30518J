#!/usr/bin/env python
# coding: utf-8

# In[58]:


from queue import PriorityQueue

def bfs(graph, start, goal, heuristics):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristics[start], start))
    path = {start : [start]}
    
    while not pq.empty():
        cost, node = pq.get()
        
        if node == goal:
            print("goal reached:", node)
            print("path", "->".join(path[node]))
            return  # Return inside the function
        if node in visited:
            continue
        visited.add(node)
        print("visiting node:", node)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                pq.put((heuristics[neighbor], neighbor))
                path[neighbor]=path[node]+[neighbor]
    print("goal not reachable")

graph = {
    'A':[('B', 4), ('C', 3)],
    'B':[('C', 1), ('D', 5)],
    'C':[('D', 2)],
    'D':[('E', 3)],
    'E':[]
}
heuristics = {
    'A':7,
    'B':6,
    'C':2,
    'D':1,
    'E':0
}
bfs(graph, 'A', 'E', heuristics)


# In[ ]:





# In[ ]:




