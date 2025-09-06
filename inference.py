import random
import time
import numpy as np # Added for tensor operations
import torch
from torch import nn, optim, distributions
from torch.nn import functional as F
import json 



class EYES(nn.Module):
    def __init__(self, num_actions, img_width, img_height):
        super().__init__()

        """
        [30][30][1]
        conv2d( kernal size 3 , 16 feature planes )
        [30][30][16]
        maxpooling(kernal size 3)
        [10][10][16]
        ReLU()

        conv2d( kernal size 3 , 16 feature planes )
        maxpooling(kernal size 3)
        [3][3][16]
        ReLU()

        linear ( 3 * 3 * 16 , 3)

        """

        self.action_space = num_actions

        h = img_height
        w = img_width

        self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1 )
        self.pool1 = nn.MaxPool2d(kernel_size=3)

        w //= 3
        h //= 3

        self.c2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1 )
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        
        w //= 3
        h //= 3

        self.output = nn.Linear(w * h * 16, 3)


    def forward(self, x):
        batch_size = x.size(0)

        x = self.c1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.c2(x)
        x = self.pool2(x)
        x = F.relu(x)


        x = x.view(batch_size, -1)
        x = self.output(x)

        return x
 

# Game settings
GRID_WIDTH = 30
GRID_HEIGHT = 30

ENTITY_SIZE = 4 # Player and Boss size (e.g., 4x4 characters)
MAX_TURNS = 200 # Maximum turns allowed per episode

SPACE_CHAR = '*'
ENEMY_CHAR = '#' # Boss character
PLAYER_CHAR = '@'
BOSS_LASER_CHAR = '|' # Character to represent the boss's laser (inactive)
PLAYER_LASER_CHAR = '!' # Changed for column visibility

# Action Space: 0 = Shoot, 1 = Move Left, 2 = Move Right
ACTION_SHOOT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2

# Rendering setting
RENDER_INTERVAL = 0.05 # Time in seconds to pause between rendering frames
INTER_EPISODE_PAUSE = 5 # Time in seconds to pause between episodes

class BossShooterEnv:
    """
    A simple text-based boss shooter environment for reinforcement learning.
    Observation: 3D Tensor (num_channels, GRID_HEIGHT, GRID_WIDTH)
                 Channel 0: Player, Channel 1: Boss, Channel 2: Player Fire
    Action Space: 0 (Shoot), 1 (Move Left), 2 (Move Right)
    """
    def __init__(self):
        # self.observation_space_size = 4 # Removed, observation is now a tensor
        self.action_space = [ACTION_SHOOT, ACTION_LEFT, ACTION_RIGHT]
        self.num_obs_channels = 1 # For tensor observation

        # Game state variables
        self.player_x = None
        self.player_health = None # Player health currently not depleted by game mechanics
        self.boss_x = None
        self.boss_health = None
        self.game_over = None
        self.win_condition = None
        self.turn_counter = None
        
        self.boss_laser_column = -1 # Boss doesn't shoot

        # For player's column attack visualization & tensor
        self.player_fired_column_this_turn = -1 
        
        self.reset()

    def create_empty_grid(self): # For text rendering
        grid = []
        for _ in range(GRID_HEIGHT):
            grid.append([SPACE_CHAR] * GRID_WIDTH)
        return grid

    def place_entity(self, grid, x, y, char, size): # For text rendering
        if 0 <= x <= GRID_WIDTH - size and 0 <= y <= GRID_HEIGHT - size:
            for row_offset in range(size):
                for col_offset in range(size):
                    if 0 <= y + row_offset < GRID_HEIGHT and 0 <= x + col_offset < GRID_WIDTH:
                        grid[y + row_offset][x + col_offset] = char

    def generate_grid_for_drawing(self): # For text rendering
        current_grid = self.create_empty_grid()
        self.place_entity(current_grid, self.boss_x, 0, ENEMY_CHAR, ENTITY_SIZE)
        self.place_entity(current_grid, self.player_x, GRID_HEIGHT - ENTITY_SIZE, PLAYER_CHAR, ENTITY_SIZE)

        if self.player_fired_column_this_turn != -1:
            laser_x = self.player_fired_column_this_turn
            laser_draw_start_y = GRID_HEIGHT - ENTITY_SIZE - 1 
            laser_draw_end_y = ENTITY_SIZE 
            
            for y_coord in range(laser_draw_start_y, laser_draw_end_y - 1, -1):
                if 0 <= y_coord < GRID_HEIGHT and 0 <= laser_x < GRID_WIDTH:
                    current_grid[y_coord][laser_x] = PLAYER_LASER_CHAR
        
        if self.boss_laser_column != -1: # Inactive
            laser_start_y = ENTITY_SIZE
            laser_end_y = GRID_HEIGHT - ENTITY_SIZE
            laser_x = self.boss_laser_column
            if 0 <= laser_x < GRID_WIDTH:
                for y_coord in range(laser_start_y, laser_end_y):
                    if 0 <= y_coord < GRID_HEIGHT:
                        current_grid[y_coord][laser_x] = BOSS_LASER_CHAR
        return current_grid

    def render(self): # Text-based rendering
        grid = self.generate_grid_for_drawing()
        print('\n' * 50)
        for row in grid:
            print("".join(row))
        print(f"Boss Health: {self.boss_health}")
        print(f"Player Health: {self.player_health}")
        print(f"Turn: {self.turn_counter} / {MAX_TURNS}")
        if self.game_over:
            if self.win_condition:
                print("\nY O U   W O N !")
            else:
                if self.player_health <= 0: 
                    print("\nG A M E   O V E R (Player Defeated)")
                else: 
                    print("\nG A M E   O V E R (Time Limit Reached)")
        time.sleep(RENDER_INTERVAL)

    def update_game_logic(self):
        if self.game_over:
            return
        self.turn_counter += 1

        # --- Highlight Start: Boss Movement Logic ---
        # Randomly decide to move left, right, or stay (optional: add stay)
        move_direction = random.choice([-1, 1]) # -1 for left, 1 for right

        # Calculate potential new position
        new_boss_x = self.boss_x + move_direction

        # Apply movement, clamping to bounds
        self.boss_x = max(0, min(new_boss_x, GRID_WIDTH - ENTITY_SIZE))
        # --- Highlight End: Boss Movement Logic ---

    def step(self, action):
        if self.game_over:
            # Return current observation (tensor) and 0 reward if already over
            return self._get_observation(), 0.0, True, {}, None 

        self.player_fired_column_this_turn = -1 # Reset before processing action

        if action == ACTION_LEFT:
            self.player_x = max(0, self.player_x - 1)
        elif action == ACTION_RIGHT:
            self.player_x = min(GRID_WIDTH - ENTITY_SIZE, self.player_x + 1)
        elif action == ACTION_SHOOT:
            fire_column = self.player_x + ENTITY_SIZE // 2
            self.player_fired_column_this_turn = fire_column 

            if fire_column >= self.boss_x and fire_column < (self.boss_x + ENTITY_SIZE):
                self.boss_health -= 1 

        
        self.update_game_logic()

        reward = 1 # Base reward for taking a step
        result = None # For custom reward handling in the loop
        done = False

        if self.boss_health <= 0:
            self.boss_health = 0 
            self.game_over = True
            self.win_condition = True
            done = True
            result = 1 # Win result
        # Player health condition commented out in user's code, effectively unreachable
        # elif self.player_health <= 0: 
        #     ...
        elif self.turn_counter >= MAX_TURNS:
            self.game_over = True
            self.win_condition = False 
            done = True
            result = 0 # Loss by timeout result
        # else: game continues, reward remains 1, result remains None

        observation = self._get_observation() # This now returns the tensor
        info = {}
        return observation, reward, done, info, result

    def _get_grid_as_tensor(self):
        grid_tensor = np.zeros((self.num_obs_channels, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

        # Channel 0: Player => 1.0
        player_y_start = GRID_HEIGHT - ENTITY_SIZE
        for r_offset in range(ENTITY_SIZE):
            for c_offset in range(ENTITY_SIZE):
                r, c = player_y_start + r_offset, self.player_x + c_offset
                if 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH:
                    grid_tensor[0, r, c] = 1.0

        # Channel 1: Boss => 2.0
        boss_y_start = 0
        for r_offset in range(ENTITY_SIZE):
            for c_offset in range(ENTITY_SIZE):
                r, c = boss_y_start + r_offset, self.boss_x + c_offset
                if 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH:
                    grid_tensor[0, r, c] = 2.0
                    
        # Channel 2: Player Fire Column => 3.0
        if self.player_fired_column_this_turn != -1:
            laser_x = self.player_fired_column_this_turn
            laser_draw_start_y = GRID_HEIGHT - ENTITY_SIZE - 1 
            laser_draw_end_y = ENTITY_SIZE 
            
            for y_coord in range(laser_draw_start_y, laser_draw_end_y - 1, -1):
                if 0 <= y_coord < GRID_HEIGHT and 0 <= laser_x < GRID_WIDTH:
                    grid_tensor[0, y_coord, laser_x] = 3.0
                    
        return grid_tensor

    def _get_observation(self):
        # Returns the game state as a tensor for CNNs
        return self._get_grid_as_tensor()

    def reset(self):
        self.player_x = GRID_WIDTH // 2 - ENTITY_SIZE // 2
        self.player_health = 10 

        # Corrected boss random position:
        # self.boss_x should be between 0 and GRID_WIDTH - ENTITY_SIZE inclusive
        self.boss_x = random.randint(0, GRID_WIDTH - ENTITY_SIZE)
        
        self.boss_health = 1 

        self.game_over = False
        self.win_condition = False
        self.turn_counter = 0 
        self.boss_laser_column = -1 
        self.player_fired_column_this_turn = -1 

        return self._get_observation() # This now returns the tensor

# --- Example RL Loop using the environment ---


print("Running episodes with random actions and rendering...")
print(f"Max turns per episode: {MAX_TURNS}")
print("Actions: 0=Shoot, 1=Left, 2=Right")
print("Reward System (handled in loop based on 'result'):")
print("  - Each step base reward: +1")
print("  - WIN (result=1): Base rewards + 50 (Total T+50)")
print("  - TIMEOUT (result=0): Base rewards sum turned negative, then -30 (Total -T-30)")

from tqdm import tqdm

env = BossShooterEnv()

# model_path = 'own-model-doom_good.pth'
# model_path = 'own-model-doom-20k.pth'
model_path = 'models/own-model-doom-50k.pth'

model = EYES(img_height=GRID_WIDTH , img_width=GRID_HEIGHT, num_actions=3)
model.load_state_dict(torch.load(model_path))
model.eval()

print("Model Loaded.")
# print(model)


time.sleep(2) 
episodes_to_run = 5

# RENDER_GAME = False
RENDER_GAME = True

for episode in tqdm(range(episodes_to_run)):
    observation = env.reset() # observation is now a tensor
    done = False
    total_reward = 0
    steps_in_episode = 0
    action_log_probs = []

    print(f"\n--- Episode {episode + 1} ---")
    if RENDER_GAME:
        env.render() 

    while not done:
        
        observation_tensor = torch.from_numpy(observation)

        with torch.no_grad():
            action_logits = model(observation_tensor)
        
        action_probs = F.softmax(action_logits) # Apply softmax along the action dimension

        m = distributions.Categorical(action_probs)
        action = m.sample()
        action = action.item()
        

        # observation is a tensor, reward is base per-step, result for terminal state
        observation, reward, done, info, result = env.step(action) 
        
        if RENDER_GAME: 
            env.render()

        if result is not None: # Terminal state reached
            # print(f"Result code from step: {result}") # For debugging
            if result == 1: # Win
                total_reward = -total_reward # Makes current sum of rewards negative
                total_reward += 50

                
            elif result == 0: # Loss by timeout
                # This logic: total_reward (which is T*1) becomes -T, then -T-30
                total_reward = -total_reward # Makes current sum of rewards negative
                total_reward -= 30 

    
    total_reward_ = total_reward / 10

    print("\n--- Episode Summary ---")
    print(f"Total reward: {total_reward} {total_reward_}")
    print(f"Steps in episode: {steps_in_episode}")
    print(f"Total turns: {env.turn_counter}")

    print('episode ', episode, ' total reward' , total_reward )

    if RENDER_GAME:
        time.sleep(2)


torch.save(model.state_dict(), 'own-model-doom.pth')
print("\nAll episodes finished.")