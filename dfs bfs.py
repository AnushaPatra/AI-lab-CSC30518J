#!/usr/bin/env python
# coding: utf-8

# In[21]:


class treenode:
    def __init__(self, value):
        self.value = value
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)
        
def dfs(root):
    if not root:
        return
    print(root.value)
    for child in root.children:
        dfs(child)
        
def bfs(root):
    if not root:
        return
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.value)
        for child in node.children:
            queue.append(child)
            
root = treenode(1)
node2 = treenode(2)
node3 = treenode(3)
node4 = treenode(4)
node5 = treenode(5)
node6 = treenode(6)
node7 = treenode(7)

root.add_child(node2)
root.add_child(node3)
root.add_child(node4)
node2.add_child(node5)
node2.add_child(node6)
node4.add_child(node7)

print("dfs:")
dfs(root)
print("\nbfs:")
bfs(root)


# In[ ]:





# In[ ]:




