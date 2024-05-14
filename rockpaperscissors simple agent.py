#!/usr/bin/env python
# coding: utf-8

# In[43]:


import random

class RockPaperScissors:
    def __init__(self):
        self.choices = ['rock', 'paper', 'scissors']
    
    def play_round(self, user_choice):
        computer_choice = random.choice(self.choices)
        
        if user_choice not in self.choices:
            return "Invalid choice! Please choose rock, paper, or scissors."
        
        print("Computer chose:", computer_choice)
        
        if user_choice == computer_choice:
            return "It's a tie!"
        elif (user_choice == 'rock' and computer_choice == 'scissors') or              (user_choice == 'paper' and computer_choice == 'rock') or              (user_choice == 'scissors' and computer_choice == 'paper'):
            return "You win!"
        else:
            return "Computer wins!"

# Example usage
game = RockPaperScissors()
user_choice = input("Enter your choice (rock, paper, or scissors): ").lower()
result = game.play_round(user_choice)
print(result)


# In[ ]:




