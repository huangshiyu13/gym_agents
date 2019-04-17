import init_dirs

import TARTRL.gym as gym

from TART.utils import get_cpu

game_name = 'Pong-v0'
env = gym.make(game_name)

action_size = env.action_size
save_model_path ='./weights/init.pkl'
hist_size = 1
input_size = 6400