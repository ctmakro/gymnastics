from __future__ import print_function
import numpy as np
import gym
from gym.spaces import Discrete, Box

# DQN for Reinforcement Learning
# by Qin Yongliang
# 2017 01 11

# run episode with some policy of some agent, and collect the rewards
# well the usual gym stuff
def do_episode_collect_trajectory(agent, env, max_steps, render=True, feed=True, realtime=False, use_best=False):
    observation = env.reset() # s1
    agent.wakeup() # notify the agent episode starts
    total_reward=0
    for t in range(max_steps):
        action = agent.act(observation,use_best=use_best) # a1

        old_observation = observation

        # s2, r1,
        observation, reward, done, _info = env.step(action)

        # d1
        isdone = 1 if done else 0
        total_reward += reward

        if feed:
            agent.feed_immediate_data((old_observation,action,reward,isdone))

        if render and (t%10==0 or realtime==True): env.render()


        if done :
            print('episode done in',t,'steps, total reward',total_reward)
            break
    return

# keras boilerplate: the simplest way to neural networking
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras
from math import *
import keras.backend as K
import time

# our neural network agent.
class nnagent(object):
    def __init__(self, num_of_actions, num_of_observations, discount_factor,  optimizer, epsilon=-1,):
        # agent database
        self.observations = np.zeros((0,num_of_observations))
        self.actions = np.zeros((0,num_of_actions))
        self.rewards = np.zeros((0,1))
        self.isdone = np.zeros((0,1))

        # agent property
        self.num_of_actions = num_of_actions
        self.num_of_observations = num_of_observations
        self.discount_factor = discount_factor

        self.epsilon = epsilon # epsilon-greedy per David Silver's Lecture and DeepMind paper.

        # -----------------------------
        # Deep-Q-Network
        def residual(i,dimin,dimout):
            ident = Dense(dimout)(i) if dimin!=dimout else i

            i = BatchNormalization()(i)
            i = Activation('elu')(i)
            i = Dense(dimout)(i)

            return merge([ident,i],mode='sum')

        input_shape = num_of_observations
        inp = Input(shape=(input_shape,))
        i = inp
        i = residual(i,input_shape,8)
        i = residual(i,8,8)
        i = residual(i,8,8)
        i = residual(i,8,num_of_actions)
        # out = Activation('softmax')(i)
        out = i
        qfunc = Model(input=inp,output=out)
        self.qfunc = qfunc

        # ------------------------------

        # ------------------------------
        # DQN trainer
        s1 = Input(shape=(input_shape,))
        a1 = Input(shape=(num_of_actions,))
        r1 = Input(shape=(1,))
        isdone = Input(shape=(1,))
        s2 = Input(shape=(input_shape,))

        q_prediction = qfunc(s1)
        # the q values you predicted for the given state.

        def calc_target(x):
            qs2 = x[0] # q value of next state
            r1 = x[1]
            isdone = x[2]
            return K.max(qs2,axis=-1,keepdims=True) * discount_factor * (1-isdone) + r1

        q_target = merge([qfunc(s2),r1,isdone],
        mode=calc_target,output_shape=(1,))
        # target = sum of [immediate reward after action a] and [q values predicted for next state, discounted]. target is a better approximation of q function for current state, so we use it as the training target.

        # but this is only a better approximation for that action taken (a), not all possible actions. therefore we need to mask q_target.

        def mask(x):
            targ = x[0]
            pred = x[1]
            a = x[2]
            return a*targ + (1-a)*pred

        q_target = merge([q_target,q_prediction,a1],
        mode=mask,output_shape=(self.num_of_actions,),name='q_target')
        # what we meant: q_target = a * q_target + (1-a) * q_prediction

        def mse(x):
            return K.mean((x[0] - x[1])**2, axis=-1,keepdims=True)

        q_loss = merge([q_target,q_prediction],
        mode=mse,output_shape=(1,),name='q_loss')
        # what we meant: q_loss = (q_target - q_prediction)**2

        qtrain = Model(input=[s1,a1,r1,isdone,s2],output=q_loss)

        def pass_thru(y_true,y_pred):
            return K.mean(y_pred)

        qtrain.compile(loss=pass_thru,optimizer=optimizer)

        # -----------------------------
        self.qfunc = qfunc
        self.qtrain = qtrain

        print('agent Initialized with',num_of_observations,'dim input and',num_of_actions,'dim output.')
        print('discount_factor',discount_factor)
        print('model architechture:')
        qfunc.summary()
        print('trainer architechture:')
        qtrain.summary()

    # act one step base on observation
    def act(self, observation, use_best=False):
        qfunc = self.qfunc
        epsilon = self.epsilon # greedy factor

        observation = observation.reshape((1,len(observation)))

        # observation is a vector
        qvalues = qfunc.predict([observation])[0]

        # for qfunc:
        # with probability epsilon we act randomly:
        if self.epsilon > np.random.rand(1) and use_best==False:
            action_index = np.random.choice(len(qvalues))
        else:
            # with probability 1-epsilon we act greedy:
            action_index = qvalues.argmax()

        return action_index

    def wakeup(self):
        # clear states
        pass

    # after playing for one(or whatever) episode, we could feed the agent with data.
    def feed_episodic_data(self,episodic_data):
        observations,actions,rewards,isdone = episodic_data

        actions = np.array(actions)
        rewards = np.array(rewards).reshape((-1,1))
        isdone = np.array(isdone).reshape((-1,1))

        # IMPORTANT: convert actions to their one-hot representations
        def one_hot(tensor,classes):
            heat = np.zeros(tensor.shape+(classes,))
            for i in range(classes):
                heat[...,i] = tensor[...] == i
            return heat
        onehot_actions = one_hot(actions,self.num_of_actions)

        # add to agent's database
        self.observations = np.vstack((self.observations,observations))
        self.actions = np.vstack((self.actions,onehot_actions))
        self.rewards = np.vstack((self.rewards,rewards))
        self.isdone = np.vstack((self.isdone,isdone))

    def feed_immediate_data(self,immediate_data):
        observation,action,rewards,isdone = immediate_data

        action = np.array(action)
        reward = np.array(rewards).reshape((-1,1))
        isdone = np.array(isdone).reshape((-1,1))

        # IMPORTANT: convert actions to their one-hot representations
        def one_hot(tensor,classes):
            heat = np.zeros(tensor.shape+(classes,))
            for i in range(classes):
                heat[...,i] = tensor[...] == i
            return heat
        onehot_action = one_hot(action,self.num_of_actions)

        # add to agent's database
        self.observations = np.vstack((self.observations,observation))
        self.actions = np.vstack((self.actions,onehot_action))
        self.rewards = np.vstack((self.rewards,reward))
        self.isdone = np.vstack((self.isdone,isdone))

        # self_train
        if len(self.observations)>50:
            self.train(epochs=1)

    # train agent with some of its collected data from its database
    def train(self,epochs=10):
        qtrain = self.qtrain
        observations,actions,rewards,isdone = self.observations, self.actions,self.rewards,self.isdone
        length = len(observations)

        for i in range(epochs):
            # train 1 epoch on a randomly selected subset of the whole database.

            subset_size = min(length-1,512)

            indices = np.random.choice(length-1,subset_size,replace=False)

            subset_observations = np.take(observations,indices,axis=0)
            subset_actions = np.take(actions,indices,axis=0)
            subset_rewards = np.take(rewards,indices,axis=0)
            subset_isdone = np.take(isdone,indices,axis=0)
            subset_next_observations = np.take(observations,indices+1,axis=0)

            qtrain.fit([
            subset_observations,
            subset_actions,
            subset_rewards,
            subset_isdone,
            subset_next_observations
            ], np.random.rand(subset_size),
                      batch_size=subset_size,
                      nb_epoch=1,
                      shuffle=False)

from gym import wrappers

# give it a try
# env = gym.make('Acrobot-v1')
env = gym.make('CartPole-v1')
# env = wrappers.Monitor(env,'./experiment-3',force=True)
agent = nnagent(
num_of_actions=env.action_space.n,
num_of_observations=env.observation_space.shape[0],
discount_factor=.95,
epsilon=1.,
optimizer = SGD(lr=0.01,momentum=0.95,nesterov=True)
)

# main training loop
def r(times=3):
    for k in range(times):
        print('training loop',k,'/',times)
        for i in range(1): # do 1 episode
            print('play episode:',i)
            do_episode_collect_trajectory(agent,env,max_steps=5000,render=True,feed=True)
            # after play, the episodic data will be feeded to the agent AUTOMATICALLY, so no feeding here

        # wait until collected data became diverse enough
        if len(agent.observations)> 50:

            # ask agent to train itself, with previously collected data
            # agent.train(epochs=min(100,len(agent.observations)))

            # decrease epsilon to make agent choose less and less random actions.
            agent.epsilon *= 0.95
            agent.epsilon = max(0.02,agent.epsilon)
            print('agent epsilon:', agent.epsilon)

def check():
    do_episode_collect_trajectory(agent,env,max_steps=1000,render=True,feed=False,realtime=True,use_best=True)
