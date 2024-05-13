#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random

def print_board(board):
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i+3]))
        if i < 6:
            print("-" * 9)

def check_winner(board, player):
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for combo in winning_combinations:
        if all(board[pos] == player for pos in combo):
            return True
    return False

def get_empty_positions(board):
    return [str(i+1) for i in range(9) if board[i] == ' ']

def player_move(board, player):
    while True:
        print("Available moves:", get_empty_positions(board))
        choice = input("Enter a number to place your mark: ")
        if choice in get_empty_positions(board):
            board[int(choice) - 1] = player
            break
        else:
            print("Invalid move! Try again.")

def agent_move(board, player):
    empty_positions = get_empty_positions(board)
    if empty_positions:
        position = random.choice(empty_positions)
        board[int(position) - 1] = player

def play_game():
    board = [' ' for _ in range(9)]
    players = ['X', 'O']
    human_player = input("Choose your symbol (X or O): ").upper()
    if human_player not in players:
        print("Invalid symbol! Please choose either 'X' or 'O'.")
        return
    computer_player = 'X' if human_player == 'O' else 'O'
    current_player = random.choice(players)
    while True:
        print_board(board)
        if check_winner(board, human_player):
            print("You win!")
            break
        elif check_winner(board, computer_player):
            print("Computer wins!")
            break
        elif all(cell != ' ' for cell in board):
            print("It's a draw!")
            break
        if current_player == human_player:
            player_move(board, current_player)
        else:
            agent_move(board, current_player)
        current_player = human_player if current_player == computer_player else computer_player

play_game()


# In[4]:



from itertools import permutations

def solve_cryptarithmetic(puzzle):
   # Extract unique letters from the puzzle
    letters = set([char for char in puzzle if char.isalpha()])
    if len(letters) > 10:
        return "Too many letters for a valid solution"

    # Generate all permutations of digits for the letters in the puzzle
    for perm in permutations('0123456789', len(letters)):
        sol = dict(zip(letters, perm))
        # Check for leading zero in any of the numbers
        if any(sol[word[0]] == '0' for word in puzzle.replace('=', '==').split() if word.isalpha()):
            continue
        # Replace letters with digits and evaluate the puzzle
        try:
            # Construct the equation by replacing letters with corresponding digits
            equation = puzzle.translate(str.maketrans(sol))
            # Check if the current permutation solves the puzzle
            if eval(equation):
                return sol
        except ArithmeticError:
            pass

    return "No solution found"

# Define the puzzle
puzzle = "EAT + THAT == APPLE"

# Solve the puzzle
solution = solve_cryptarithmetic(puzzle)

# Print the solution
if isinstance(solution, dict):
    print("Solution:")
    for letter, digit in solution.items():
        print(f"{letter} = {digit}")
else:
    print(solution)


# In[5]:


class TreeNode:
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

# Example tree
#       1
#     / | \
#    2  3  4
#   / \    |
#  5   6   7

root = TreeNode(1)
node2 = TreeNode(2)
node3 = TreeNode(3)
node4 = TreeNode(4)
node5 = TreeNode(5)
node6 = TreeNode(6)
node7 = TreeNode(7)

root.add_child(node2)
root.add_child(node3)
root.add_child(node4)
node2.add_child(node5)
node2.add_child(node6)
node4.add_child(node7)

print("DFS traversal:")
dfs(root)
print("\nBFS traversal:")
bfs(root)


# In[8]:


from queue import PriorityQueue

def best_first_search(graph, start, goal, heuristic):
    visited = set()
    priority_queue = PriorityQueue()
    priority_queue.put((heuristic[start], start))
    path = {start: [start]}  # To keep track of the path

    while not priority_queue.empty():
        cost, node = priority_queue.get()

        if node == goal:
            print("Goal reached:", node)
            print("Path:", "->".join(path[node]))
            return

        if node in visited:
            continue

        visited.add(node)

        print("Visiting node:", node)

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                priority_queue.put((heuristic[neighbor], neighbor))
                path[neighbor] = path[node] + [neighbor]  # Update the path for the neighbor

    print("Goal not reachable")

# Example graph
graph = {
    'A': [('B', 4), ('C', 3)],
    'B': [('C', 1), ('D', 5)],
    'C': [('D', 2)],
    'D': [('E', 3)],
    'E': []
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 1,
    'E': 0
}

best_first_search(graph, 'A', 'E', heuristic)


# In[9]:


##### from queue import PriorityQueue

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


# In[10]:


class FuzzySystem:
    def __init__(self, rules):
        self.rules = rules

    def infer(self, input_value):
        output_value = 0
        total_weight = 0

        for rule in self.rules:
            weight = rule['function'](input_value)
            output_value += weight * rule['consequence']
            total_weight += weight

        return output_value / total_weight if total_weight != 0 else 0

# Example fuzzy rules
rules = [
    {'function': lambda x: 1 if x <= 50 else 0, 'consequence': 0},
    {'function': lambda x: (100 - x) / 50 if 50 < x <= 100 else 0, 'consequence': 0.5},
    {'function': lambda x: 1 if x > 100 else 0, 'consequence': 1}
]

# Create fuzzy system
fuzzy_system = FuzzySystem(rules)

# Example input value (ambient light intensity)
input_value = 75

# Infer output (brightness level)
output_value = fuzzy_system.infer(input_value)

print("Input (Ambient light intensity):", input_value)
print("Output (Brightness level):", output_value)


# In[34]:


def unify(var, x, theta):
    if theta is None:
        return None
    elif var == x:
        return theta
    elif isinstance(var, str) and var[0].islower():
        return unify_var(var, x, theta)
    elif isinstance(x, str) and x[0].islower():
        return unify_var(x, var, theta)
    elif isinstance(var, list) and isinstance(x, list) and len(var) == len(x):
        return unify(var[1:], x[1:], unify(var[0], x[0], theta))
    else:
        return None

def unify_var(var, x, theta):
    if var in theta:
        return unify(theta[var], x, theta)
    elif x in theta:
        return unify(var, theta[x], theta)
    else:
        theta[var] = x
        return theta

# Example usage
theta = unify(['?x', '?y', '?z'], [1, 2, 3], {})
print("Unification result:", theta)  # Output: None


# In[35]:


def resolution(clause1, clause2):
    resolvents = []
    for literal1 in clause1:
        for literal2 in clause2:
            if literal1 == -literal2:
                resolvent = set(clause1).union(set(clause2))
                resolvent.remove(literal1)
                resolvent.remove(literal2)
                resolvents.append(list(resolvent))
    return resolvents

# Example usage
clause1 = [1, 2, 3]
clause2 = [-2, 4]
result = resolution(clause1, clause2)
print("Resolvent:", result)


# In[20]:


import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.intercept = None
        self.slope = None

    def fit(self, X, y):
        self.slope, self.intercept = np.polyfit(X.flatten(), y, 1)

    def predict(self, X):
        return self.slope * X + self.intercept

# Example usage
X = np.array([1, 2, 3])
y = np.array([2, 4, 6])

model = SimpleLinearRegression()
model.fit(X, y)

new_X = np.array([4, 5])
predictions = model.predict(new_X)
print("Predictions:", predictions)


# In[23]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# Example usage
text = "I love this product! It's amazing."
sentiment_score = analyze_sentiment(text)
print("Sentiment Score:", sentiment_score)


# In[22]:


import nltk
nltk.download('vader_lexicon')


# In[24]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Generate some sample data
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, size=(100,))  # Binary labels

# Define the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[ ]:




