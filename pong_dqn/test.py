import init_dirs
import TARTRL.gym as gym
from parameters import *
import tensorflow as tf
from model import DQN_Model as Model
from TART.model import Saver
import numpy as np
import time
if __name__ == '__main__':
    sess = tf.Session()

    inputs_sequence_holder = tf.placeholder(tf.float32,
                                            [1, input_size * hist_size],
                                            name="inputs")

    model = Model(name='test')

    model.build_actor(inputs=inputs_sequence_holder,
                      action_size=action_size)

    sess.run(tf.global_variables_initializer())
    saver = Saver(sess)

    saver.load(save_model_path, model, del_scope=True, strict=True)

    env = gym.make(game_name)

    while True:
        state = env.reset()
        rewards_sum = 0
        done = False
        step_now = 0

        while not done:
            feeds = {inputs_sequence_holder: state.reshape(1, input_size * hist_size)
                     }
            Qs = sess.run(
                model.Qs, feed_dict=feeds)
            action = np.argmax(Qs)

            env.render()
            obs, reward, done, info = env.step(action)

            state = obs
            rewards_sum += reward
            step_now += 1
            time.sleep(0.02)
        print(rewards_sum)
