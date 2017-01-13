# well..

# gym boilerplate
from __future__ import print_function
import numpy as np
import gym
from gym.spaces import Discrete, Box

# keras boilerplate: the simplest way to neural networking
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras
from math import *
import random
import keras.backend as K
import time

from collections import deque

# replay buffer per http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
class rpm(object):
    #replay memory
    def __init__(self,size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, tup):
        experience = tup
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        item_count = len(batch[0])
        res = []
        for i in range(item_count):
            res.append(np.array([item[0] for item in batch]))
        return res

# residual dense unit
def resdense(features):
    def unit(i):
        hfeatures = max(4,int(features/4))

        ident = i
        i = Dense(features,activation='tanh')(i)

        ident = Dense(hfeatures)(ident)
        ident = Dense(features)(ident)

        return merge([ident,i],mode='sum')
    return unit

class nnagent(object):
    def __init__(self,
    observation_space,
    action_space,
    discount_factor, # gamma
    optimizer
    ):
        self.rpm = rpm(100000) # 100k history

        self.inputdims = observation_space.shape[0]
        # assume observation_space is continuous

        if isinstance(action_space,Box): # if action space is continuous

            low = action_space.low
            high = action_space.high

            num_of_actions = action_space.shape[0]

            self.action_bias = (high+low)/2.
            self.action_multiplier = high - self.action_bias

            # say high,low -> [2,7], then bias -> 4.5
            # mult = 2.5. then [-1,1] multiplies 2.5 + bias 4.5 -> [2,7]

            self.is_continuous = True
        else:
            num_of_actions = action_space.n

            self.action_bias = .5
            self.action_multiplier = .5 # map (-1,1) into (0,1)

            self.is_continuous = False

        self.outputdims = num_of_actions

        self.discount_factor = discount_factor
        self.optimizer = optimizer

        ids,ods = self.inputdims,self.outputdims
        self.actor = self.create_actor_network(ids,ods)
        self.critic = self.create_critic_network(ids,ods)

        print('inputdims:{}, outputdims:{}'.format(self.inputdims,self.outputdims))
        print('actor network:')
        self.actor.summary()
        print('critic network:')
        self.critic.summary()

        # identical copy of actor and critic
        self.actor_target = self.create_actor_network(ids,ods)
        self.critic_target = self.create_critic_network(ids,ods)


    # replace target networks with online network weights
    def replace_weights(self):
        theta_a,theta_c = self.actor.get_weights(),self.critic.get_weights()
        theta_a_targ,theta_c_targ = self.actor_target.get_weights(),self.critic_target.get_weights()

        tau = 0.01 # mixing factor: we dont entirely replace
        theta_a_targ = [theta_a[i]*tau + theta_a_targ[i]*(1-tau) for i in range(len(theta_a))]
        theta_c_targ = [theta_c[i]*tau + theta_c_targ[i]*(1-tau) for i in range(len(theta_c))]

        self.actor_target.set_weights(theta_a_targ)
        self.critic_target.set_weights(theta_c_targ)

    # a = actor(s)
    def create_actor_network(self,inputdims,outputdims):
        inp = Input(shape=(inputdims,))
        i = inp
        i = resdense(32)(i)
        i = resdense(32)(i)
        i = resdense(64)(i)
        i = resdense(outputdims)(i)
        # map into (0,1)
        i = Activation('tanh')(i)
        # map into action_space
        i = Lambda(lambda x:x * self.action_multiplier + self.action_bias)(i)

        out = i
        model = Model(input=inp,output=out)
        model.compile(loss='mse',optimizer=self.optimizer)
        return model

    # q = critic(s,a)
    def create_critic_network(self,inputdims,actiondims):
        inp = Input(shape=(inputdims,))
        act = Input(shape=(actiondims,))
        i = merge([inp,act],mode='concat')

        i = resdense(64)(i)
        i = resdense(32)(i)
        i = resdense(32)(i)
        i = resdense(1)(i)
        out = i
        model = Model(input=[inp,act],output=out)
        model.compile(loss='mse',optimizer=self.optimizer)
        return model

    def train(self):
        memory = self.rpm
        critic = self.critic
        actor = self.actor

        if memory.size()>100:
            #if enough samples
            batch_size = 64
            verbose = 1

            [s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)

            # a2_targ = actor_targ(s2) : what old actor suggests for s2
            a2 = self.actor_target.predict(s2)

            # q2_targ = critic_targ(s2,a2) :  what old critic thinks of the suggestion
            q2 = self.critic_target.predict([s2,a2])

            # Bellman!
            # q1 = r1 + gamma * q2 (if not done, else r1)
            q1_target = r1 + (1-isdone) * self.discount_factor * q2

            # train the critic to better predict its predecessor.
            critic.fit([s1,a1],q1_target
            batch_size=batch_size,
            nb_epoch=1,
            verbose=verbose,
            shuffle=False
            )

            # a1_pred = actor(s1) : what current actor think should be done at s1
            a1_pred = actor.predict(s1)

        else:
            print('# no enough samples, not training')
