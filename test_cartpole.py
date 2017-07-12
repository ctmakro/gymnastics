import vrepper
from vrepper.vrepper import vrepper

import os,time
import numpy as np

import gym
from gym import spaces
# from gym.utils import colorize, seeding

import cv2
from cv2tools import filt,vis

class CartPoleVREPEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }
    def __init__(self,headless=False):
        self.venv = venv = vrepper(headless=headless)
        venv.start()
        venv.load_scene(
            os.getcwd() + '/scenes/cart_pole.ttt')

        self.slider = venv.get_object_by_name('slider')
        self.cart = venv.get_object_by_name('cart')
        self.mass = venv.get_object_by_name('mass')

        self.webcam = venv.get_object_by_name('webcam')

        print('(CartPoleVREP) initialized')

        obs = np.array([np.inf]*6)
        act = np.array([2.])

        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-obs,obs)

    def _self_observe(self):
        # observe then assign
        cartpos = self.cart.get_position()
        masspos = self.mass.get_position()
        cartvel,cart_angvel = self.cart.get_velocity()
        massvel,mass_angvel = self.mass.get_velocity()

        self.observation = np.array([
            cartpos[0],cartvel[0],
            masspos[0],masspos[2],
            massvel[0],massvel[2]
            ]).astype('float32')

    def _step(self,actions):
        # actions = np.clip(actions, -1, 1)
        assert self.action_space.contains(actions)
        v = actions[0]

        # step
        self.slider.set_velocity(v)
        self.venv.step_blocking_simulation()

        # observe again
        self._self_observe()

        # cost
        height_of_mass = self.observation[3] # masspos[2]
        cost = - height_of_mass + (v**2) * 0.001

        return self.observation, -cost, False, {}

    def _render(self, mode='human', close=False):
        if close:
            cv2.destroyAllWindows()
            return

        im = self.webcam.get_vision_image()
        cv2.imshow('render',im)
        cv2.waitKey(1)
        # print(dim,im)

    def _reset(self):
        self.venv.stop_blocking_simulation()
        self.venv.start_blocking_simulation()
        self._self_observe()
        return self.observation

    def _destroy(self):
        self.venv.stop_blocking_simulation()
        self.venv.end()

if __name__ == '__main__':
    from ddpg2 import nnagent

    env = CartPoleVREPEnv(headless=True)
    # env = CartPoleVREPEnv(headless=False)

    def test_environment():
        for k in range(5):
            observation = env.reset()
            for _ in range(20):
                env.render()
                action = env.action_space.sample() # your agent here (this takes random actions)
                observation, reward, done, info = env.step(action)
                print(reward)

        print('simulation ended. leaving in 5 seconds...')
        time.sleep(5)

    agent = nnagent(
        env.observation_space,
        env.action_space,
        discount_factor=.99,
        stack_factor=1,
        train_skip_every=1,
    )

    noise_level = .5
    def r(ep):
        agent.render = False
        agent.training = True
        global noise_level
        # agent.render = True
        e = env
        for i in range(ep):
            noise_level *= .99
            noise_level = max(3e-3, noise_level)
            print('ep',i,'/',ep,'noise_level',noise_level)
            agent.play(e,realtime=False,max_steps=20*6,noise_level=noise_level)

    def test():
        e = env
        agent.render = True
        agent.training = False
        agent.play(e,realtime=True,max_steps=20*6,noise_level=1e-11,)
