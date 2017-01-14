from __future__ import print_function

# Deep Deterministic Policy Gradient Method
# David Silver et al.

# implemented in plain Keras, by Qin Yongliang
# 2017 01 13

'''
summary

0. s for state, a for action, r for reward,
    q for 'action_quality', or expectation of sum of discounted future reward.

1. you have 2 network, Mr. actor and Mr. critic
    - Mr. actor generate actions: a = actor(s)
    - Mr. critic score (state,action) pairs: q = critic(s,a)

    >in literature, Mr. actor is function mu(s), Mr. critic is function Q(s,a)

2. you improve Mr. critic by using Bellman equation, or what they call TD-learning
    - Q(s1,a1) := r1 + gamma * Q(s2,a2) where a2 = actor(s2)
    - train Mr. critic to predict the calculated Q(s1,a1) given s1 and a1, using gradient descent and MSE loss.

3. after that, improve Mr. actor by gradient ascent w.r.t. Q(s,a)
    - a1_maybe = actor(s1), q1_maybe = critic(s1,a1_maybe)
    - therefore q1_maybe = critic(s1,actor(s1)). we want to increase q1_maybe!!
    - then figure out what is the gradient of actor w.r.t. q1_maybe,
        using tf.gradient() or by compositing Keras Models (as I did, to keep things clean)
    - then do gradient ascent to increase Mr. actor's actions' q-value

4. to stabilize the whole learning process:
    - random sampling of training examples from replay memory
    - use 'target' networks that are copy of actor and critic,
        their weights gradually shift towards the weights of the real actor and critic
        to reduce self-correlation/oscillation (well, if you know control theory)
    - add noise to actor's output in the beginning of learning, to turn deterministic actions into probabilistic ones
    - that's basically it

5. now go master the game of Gym
'''

'''
personal tricks:

check the Residual Dense Unit, it works!
'''

# gym boilerplate
import numpy as np
import gym
from gym import wrappers
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
    def __init__(self,buffer_size):
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
            k = np.array([item[i] for item in batch])
            if len(k.shape)==1: k = k.reshape(k.shape+(1,))
            res.append(k)
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
        self.rpm = rpm(1000000) # 1M history
        self.observation_stack_factor = 2

        self.inputdims = observation_space.shape[0] * self.observation_stack_factor
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

            def clamper(actions):
                return np.clip(actions,a_max=action_space.high,a_min=action_space.low)

            self.clamper = clamper
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
        self.critic, self.frozen_critic = self.create_critic_network(ids,ods)

        print('inputdims:{}, outputdims:{}'.format(ids,ods))
        print('actor network:')
        self.actor.summary()
        print('critic network:')
        self.critic.summary()

        # target networks: identical copies of actor and critic
        self.actor_target = self.create_actor_network(ids,ods)
        self.critic_target, self.frozen_critic_target = self.create_critic_network(ids,ods)

        self.replace_weights(tau=1.)

        # now the dirty part: the actor trainer --------------------------------

        # explaination of this part is written in the train() method

        s_given = Input(shape=(self.inputdims,))
        a1_maybe = self.actor(s_given)
        q1_maybe = self.frozen_critic([s_given,a1_maybe])
        # frozen weight version of critic. so we can train only the actor

        actor_trainer = Model(input=s_given,output=q1_maybe)

        # use negative of q1_maybe as loss (so we can maximize q by minimizing the loss)
        def neg_q1(y_true,y_pred):
            return - y_pred # neat!

        actor_trainer.compile(optimizer=self.optimizer,loss=neg_q1)
        self.actor_trainer = actor_trainer
        # dirty part ended -----------------------------------------------------

    # (gradually) replace target network weights with online network weights
    def replace_weights(self,tau=0.002):
        theta_a,theta_c = self.actor.get_weights(),self.critic.get_weights()
        theta_a_targ,theta_c_targ = self.actor_target.get_weights(),self.critic_target.get_weights()

        # mixing factor tau : we gradually shift the weights...
        theta_a_targ = [theta_a[i]*tau + theta_a_targ[i]*(1-tau) for i in range(len(theta_a))]
        theta_c_targ = [theta_c[i]*tau + theta_c_targ[i]*(1-tau) for i in range(len(theta_c))]

        self.actor_target.set_weights(theta_a_targ)
        self.critic_target.set_weights(theta_c_targ)

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        inp = Input(shape=(inputdims,))
        i = inp
        i = resdense(32)(i)
        i = resdense(32)(i)
        i = resdense(64)(i)
        i = resdense(64)(i)
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

    # q = critic(s,a) : predict q given state and action
    def create_critic_network(self,inputdims,actiondims):
        inp = Input(shape=(inputdims,))
        act = Input(shape=(actiondims,))
        i = merge([inp,act],mode='concat')

        i = resdense(64)(i)
        i = resdense(64)(i)
        i = resdense(64)(i)
        i = resdense(32)(i)
        i = resdense(32)(i)
        i = resdense(1)(i)
        out = i
        model = Model(input=[inp,act],output=out)
        model.compile(loss='mse',optimizer=self.optimizer)

        # now we create a frozen_model,
        # that uses the same layers with weights frozen when trained.
        for i in model.layers:
            i.trainable = False # froze the layers

        frozen_model = Model(input=[inp,act],output=out)
        frozen_model.compile(loss='mse',optimizer=self.optimizer)

        return model,frozen_model

    def train(self,verbose=1):
        memory = self.rpm
        critic,frozen_critic = self.critic,self.frozen_critic
        actor = self.actor
        batch_size = 64

        if memory.size() > batch_size:
            #if enough samples in memory

            # sample randomly a minibatch from memory
            [s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)
            # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)

            # a2_targ = actor_targ(s2) : what will you do in s2, Mr. old actor?
            a2 = self.actor_target.predict(s2)

            # q2_targ = critic_targ(s2,a2) : how good is action a2, Mr. old critic?
            q2 = self.critic_target.predict([s2,a2])

            # if a2 is q2-good, then what should q1 be?
            # Use Bellman Equation! (recursive definition of q-values)
            # if not last step of episode:
            #   q1 = (r1 + gamma * q2)
            # else:
            #   q1 = r1

            q1_target = r1 + (1-isdone) * self.discount_factor * q2
            # print(q1_target.shape)

            # train the critic to predict the q1_target, given s1 and a1.
            critic.fit([s1,a1],q1_target,
            batch_size=batch_size,
            nb_epoch=1,
            verbose=verbose,
            shuffle=False
            )

            # now the critic can predict more accurate q given s and a.
            # thanks to the Bellman equation, and David Silver.

            # with a better critic, we can now improve our actor!

            if False: # the following part is for explaination purposes

                # a1_pred = actor(s1) : what will you do in s1, Mr. actor?
                a1_maybe = actor.predict(s1)
                # this action may not be optimal. now let's ask the critic.

                # what do you think of Mr. actor's action on s1, Mr. better critic?
                q1_maybe = critic.predict([s1,a1_maybe])

                # what should we do to the actor, to increase q1_maybe?
                # well, calculate the gradient of actor parameters
                # w.r.t. q1_maybe, then do gradient ascent.

                # so let's build a model that trains the actor to output higher q1_maybe values

                s_given = Input(shape=(self.inputdims,))
                a1_maybe = actor(s_given)
                q1_maybe = frozen_critic([s_given,a1_maybe])
                # frozen weight version of critic. so we only train the actor

                actor_trainer = Model(input=s_given,output=q1_maybe)

                # use negative of q1_maybe as loss (so we can maximize q by minimizing the loss)
                def neg_q1(y_true,y_pred):
                    return - y_pred # neat!

                actor_trainer.compile(optimizer=self.optimizer,loss=neg_q1)

            else: # the actor_trainer is already initialized in __init__
                actor_trainer = self.actor_trainer

                actor_trainer.fit(s1,
                np.zeros((batch_size,1)), # useless target label
                batch_size=batch_size,
                nb_epoch=1,
                verbose=verbose,
                shuffle=False
                )

            # now both the actor and the critic have improved.
            self.replace_weights()

        else:
            pass
            # print('# no enough samples, not training')

    def feed_one(self,tup):
        self.rpm.add(tup)

    # gymnastics
    def play(self,env,max_steps=-1,realtime=False,render=True,noise_level=0.): # play 1 episode
        max_steps = max_steps if max_steps > 0 else 5000
        steps = 0
        total_reward = 0

        # stack a little history to ensure markov property
        # LSTM will definitely be used here in the future...
        global que # python 2 quirk
        que = np.zeros((self.inputdims,),dtype='float32') # list of recent history actions

        def quein(observation):
            global que # python 2 quirk
            length = que.shape[0]
            que = np.hstack([que,observation])[-length:]

        # what the agent see as state is a stack of history observations.

        observation = env.reset()
        quein(observation) # quein o1
        lastque = que.copy() # s1

        while True and steps <= max_steps:
            steps +=1

            # add noise to our actions, since our policy by nature is deterministic
            exploration_noise = np.random.normal(loc=0.,scale=noise_level,size=(self.outputdims,))

            action = self.act(lastque) # a1
            action += exploration_noise
            action = self.clamper(action)

            # o2, r1,
            observation, reward, done, _info = env.step(action)

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            quein(observation) # quein o2
            nextque = que.copy() # s2

            # feed into replay memory
            self.feed_one((lastque,action,reward,isdone,nextque)) # s1,a1,r1,isdone,s2

            lastque = nextque

            if render and (steps%10==0 or realtime==True): env.render()
            if done :
                break

            verbose= 2 if steps==1 else 0
            self.train(verbose=verbose)

        print('episode done in',steps,'steps, total reward',total_reward)
        return

    # one step of action, given observation
    def act(self,observation):
        actor = self.actor
        obs = np.reshape(observation,(1,len(observation)))
        actions = actor.predict([obs])[0]
        return actions

class playground(object):
    def __init__(self,envname):
        self.envname=envname
        env = gym.make(envname)
        self.env = env

        self.monpath = './experiment-'+self.envname

    def wrap(self):
        from gym import wrappers
        self.env = wrappers.Monitor(self.env,self.monpath,force=True)

    def up(self):
        self.env.close()
        gym.upload(self.monpath, api_key='sk_ge0PoVXsS6C5ojZ9amTkSA')

p = playground('LunarLanderContinuous-v2')
e = p.env

agent = nnagent(
e.observation_space,
e.action_space,
discount_factor=.995,
optimizer=RMSprop()
)

def r(ep):
    e = p.env
    for i in range(ep):
        noise_level = max(3e-2,(50.-i)/50.)
        print('ep',i,'/',ep,'noise_level',noise_level)
        agent.play(e,max_steps=1000,noise_level=noise_level)
