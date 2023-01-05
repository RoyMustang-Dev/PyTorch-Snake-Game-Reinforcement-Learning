import torch
import random
import numpy as np
# Deque is a data structure where we will store the memory
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Created this to store the Game and Model in memory(deque)
class Agent:
    # Defining the Steps involved
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # to control the randomness
        self.gamma = 0.9 # this is the discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if this exceeds it will automatically remove elements from left (popleft())
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)      

    # Step 1- Get the State
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x, # Meaning food is on the left
            game.food.x > game.head.x, # Meaning food is on the right
            game.food.y < game.head.y, # Meaning food is on the up
            game.food.y > game.head.y  # Meaning food is on the down
            ]

        return np.array(state, dtype=int)

    # Step 2- Remember the state and related properties
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # if this exceeds MAX_MEMORY, then popleft()

    # Step 3- Defining the Training Modeles
    def tarin_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # This will create a list of Tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Step 4- To get Action based on the states
    def get_action(self, state):
        # Do some Random Moves: this is also known as Tradeoff b/w Exploration and Exploitation in Deep Learning
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

# Starting the Train phase
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    # Starting training loop
    while True:
        # Step 1- Get Old State
        state_old = agent.get_state(game)

        # Step 2- Get the move based on the current state
        final_move = agent.get_action(state_old)

        # Step 3- Perform the move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Step 4- Train the short memory of the Agent(meaning only 1 step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Step 5- Remember all these and store it in Memory
        agent.remember(state_old, final_move, reward, state_new, done)

        # Step 6- If Done(game over), then train the long term memory, also called Replay Memory or Experience Replay
        if done:
            # Train long memory, and plot the result
            game.reset()
            agent.n_games += 1
            agent.tarin_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game :", agent.n_games, "Score :", score, "Record :", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()