from __future__ import print_function
import numpy as np
import gym
from gym.spaces import Discrete, Box

# DQN for Reinforcement Learning
# by Qin Yongliang
# 2017 01 11

def continuous_actions(env):
    if isinstance(env.action_space,Box):
        pass

    split = 7

    dims = env.action_space.shape[0]
    action_count = split * dims

    low = env.action_space.low
    high = env.action_space.high

    itvl = high - low

    global cbuf
    cbuf = np.zeros((dims),dtype='float32')

    def d2c(index):
        global cbuf
        # cbuf = cbuf*.5

        idx = index

        chosen_dim = int(idx/split)
        chosen_split = idx%split
        cbuf[chosen_dim] = chosen_split/float(split-1)

        cont = cbuf * itvl + low
        # print(cont)
        return cont

    return action_count,d2c

# run episode with some policy of some agent, and collect the rewards
# well the usual gym stuff
def do_episode_collect_trajectory(agent, env, max_steps, render=True, feed=True, realtime=False, use_best=False):
    observation = env.reset() # s1
    last_observation = observation

    agent.wakeup() # notify the agent episode starts
    total_reward=0
    for t in range(max_steps):
        global cbuf
        combined_observation = np.hstack([last_observation,observation,cbuf])
        last_observation = observation

        action = agent.act(combined_observation,use_best=use_best) # a1

        if isinstance(env.action_space,Box):
            # action_count,d2c = continuous_actions(env)
            actual_action = d2c(action)
        else:
            actual_action = action

        # s2, r1,
        observation, reward, done, _info = env.step(actual_action)

        # d1
        isdone = 1 if done else 0
        total_reward += reward

        if feed:
            agent.feed_immediate_data((combined_observation,action,reward,isdone))

        if render and (t%15==0 or realtime==True): env.render()


        if done :
            break
    print('episode done in',t,'steps, total reward',total_reward)

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

        self.big_C = 20
        self.big_C_counter = 0

        self.epsilon = epsilon # epsilon-greedy per David Silver's Lecture and DeepMind paper.

        # -----------------------------
        # Deep-Q-Network

        from keras.regularizers import l2, activity_l2

        def resdense(features):
            def unit(i):
                hfeatures = max(4,int(features/4))

                ident = i
                i = Dense(features,activation='tanh')(i)

                ident = Dense(hfeatures)(ident)
                ident = Dense(features)(ident)

                return merge([ident,i],mode='sum')
            return unit

        input_shape = num_of_observations
        inp = Input(shape=(input_shape,))
        i = inp
        # i1 = inp
        # i2 = inp


        # i = BatchNormalization()(i)
        # i = Activation('linear')(i)
        # i1 = arm(i1)
        # i2 = arm(i2)
        # i = Dense(8,activation='tanh')(i)
        # i = Dense(64,activation='relu')(i)
        # i = Dense(16,activation='tanh')(i)
        # i = Dense(1024,activation='tanh')(i)
        # i = Dense(128,activation='relu')(i)
        i = resdense(64)(i)
        i = resdense(32)(i)
        i = resdense(128)(i)
        # i = Dense(32,activation='tanh')(i)
        # i = MaxoutDense(32)(i)
        # i = MaxoutDense(32)(i)
        # i = MaxoutDense(64)(i)
        # i = MaxoutDense(8)(i)
        # i = Activation('relu')(i)
        # i = residual(i,64)
        # i = residual(i,16)
        # i = residual(i,32)
        # i = residual(i,32)
        # i = residual(i,64)
        # i = residual(i,32)
        # i = residual(i,6)

        # i = BatchNormalization()(i)

        # i = Activation('elu')(i)

        # i1 = Dense(1)(i1)
        # i2 = Dense(1)(i2)
        # i = merge([i1,i2],mode='concat')

        # i = Dense(16,activation='tanh')(i)
        # i = Dense(16,activation='tanh')(i)
        # i = Dense(16,activation='tanh')(i)
        # i = Dense(16,activation='tanh')(i)
        # i = Dense(16,activation='tanh')(i)
        # i = arm(i,128)
        # i = arm(i,16)
        # i = arm(i,16)
        # i = arm(i,16)
        # i = arm(i,32)
        # i = arm(i,num_of_actions)

        # abuf = []
        # for k in range(num_of_actions):
        #     r = Dense(5,activation='tanh')(i)
        #     r = Dense(1)(r)
        #     abuf.append(r)
        # out = merge(abuf,mode='concat')
        i = Dense(num_of_actions)(i)

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
        # s2 = Input(shape=(input_shape,))
        qs2 = Input(shape=(num_of_actions,)) # qs2 is precalc-ed

        q_prediction = qfunc(s1)
        # the q values we predicted for the given state.

        q_s1_a1 = merge([q_prediction,a1],
        mode=(lambda x:K.sum(x[0] * x[1],axis=-1,keepdims=True)),
        output_shape=(1,))

        def calc_target(x):
            qs2 = x[0] # q value of next state
            r1 = x[1]
            isdone = x[2]
            return (K.max(qs2,axis=-1,keepdims=True) * discount_factor * (1-isdone) + r1)

        q_target = merge([qs2,r1,isdone],
        mode=calc_target,output_shape=(1,))
        # target = sum of [immediate reward after action a] and [q values predicted for next state, discounted]. target is a better approximation of q function for current state, so we use it as the training target.

        def mse(x):
            return K.mean((x[0] - x[1])**2, axis=-1, keepdims=True)

        q_loss = merge([q_target,q_s1_a1],
        mode=mse,output_shape=(1,),name='q_loss')
        # what we meant: q_loss = (q_target - q_prediction)**2

        qtrain = Model(input=[s1,a1,r1,isdone,qs2],output=q_loss)

        def pass_thru(y_true,y_pred):
            return K.mean(y_pred,axis=-1)

        qtrain.compile(loss=pass_thru,optimizer=optimizer)

        # -----------------------------

        # -----------------------------
        # mirrored DQN(for 'target' calculation)
        qfunc2 = model_from_json(qfunc.to_json())

        # -------------------------

        self.qfunc = qfunc
        self.qfunc2 = qfunc2
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
        if (self.epsilon > np.random.rand(1)) and use_best==False:
            action_index = np.random.choice(len(qvalues))
        else:
            # with probability 1-epsilon we act greedy:
            action_index = qvalues.argmax()

        # print(action_index)
        from winfrey import showbar
        showbar(np.hstack([qvalues,np.max(qvalues,keepdims=True)]),action_index) #visualization

        agent.epsilon -= 1./10000
        agent.epsilon = max(0.07,agent.epsilon)


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
        if len(self.observations)>100:
            self.train(epochs=1)
            pass

    def eat(self):
        # throw away excessively long history
        length = len(self.observations)
        if length>50000:
            eat = 50000-length
            print('eating',eat,'kids..')
            self.observations = self.observations[eat:length]
            self.actions = self.actions[eat:length]
            self.rewards = self.rewards[eat:length]
            self.isdone = self.isdone[eat:length]

    # train agent with some of its collected data from its database
    def train(self,epochs=10):
        qtrain = self.qtrain
        qfunc = self.qfunc
        qfunc2 = self.qfunc2

        observations,actions,rewards,isdone = self.observations, self.actions,self.rewards,self.isdone
        length = len(observations)

        # print('----trainning for',epochs,'epochs')

        for i in range(epochs):
            # train 1 epoch on a randomly selected subset of the whole database.
            if epochs-1 == i and epochs>1:
                verbose = 2
            else:
                verbose = 0
                # muted since bad for performance.

            # before training, we may read the weights of the policy network, and load em into the 'target' network
            # do this every C steps (How DeepMind this is.)
            self.big_C_counter+=1
            self.big_C_counter%=self.big_C
            if self.big_C_counter == 0:
                thetas = qfunc.get_weights()
                qfunc2.set_weights(thetas)

            subset_size = min(length-1,128)

            indices = np.random.choice(length-1,subset_size,replace=False)

            subset_observations = np.take(observations,indices,axis=0).astype('float32')
            subset_actions = np.take(actions,indices,axis=0).astype('float32')
            subset_rewards = np.take(rewards,indices,axis=0).astype('float32')
            subset_isdone = np.take(isdone,indices,axis=0).astype('float32')

            subset_next_observations = np.take(observations,indices+1,axis=0).astype('float32')


            qs2 = self.qfunc2.predict(subset_next_observations)
            # 'target' Q-func weights are not updated every training call, to prevent potential divergence problems

            qtrain.fit([
            subset_observations,
            subset_actions,
            subset_rewards,
            subset_isdone,
            qs2
            ], np.random.rand(subset_size),
                      batch_size=subset_size,
                      nb_epoch=1,
                      verbose=verbose,
                      shuffle=False)
        # print('----done')

from gym import wrappers

# give it a try
# env = gym.make('Acrobot-v1')
# env = gym.make('Pendulum-v0')
env = gym.make('BipedalWalker-v2')
# env = gym.make('LunarLander-v2')BipedalWalker-v2
# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = wrappers.Monitor(env,'./experiment-3',force=True)

if isinstance(env.action_space,Box):
    action_count,d2c = continuous_actions(env)
    num_of_actions = action_count
else:
    num_of_actions = env.action_space.n

num_of_observations = env.observation_space.shape[0]

print('environment:',num_of_actions,'actions,',num_of_observations,'observations')

agent = nnagent(
num_of_actions=num_of_actions,
num_of_observations=num_of_observations*2 + len(cbuf),
discount_factor=.99,
epsilon=1.,
optimizer = RMSprop()
# optimizer = SGD(lr=0.0005, clipnorm=10.,momentum=0.0,nesterov=False) # momentum must = 0; use plain SGD
)

# main training loop
def r(times=3):
    for k in range(times):
        print('training loop',k,'/',times)
        for i in range(1): # do 1 episode
            print('play episode:',i)
            do_episode_collect_trajectory(agent,env,max_steps=100000,render=True,feed=True)
            # after play, the episodic data will be feeded to the agent AUTOMATICALLY, so no feeding here

        # wait until collected data became diverse enough
        if len(agent.observations)> 100:

            # ask agent to train itself, with previously collected data
            # agent.train(epochs=min(100,len(agent.observations)/4))

            # decrease epsilon to make agent choose less and less random actions.
            # agent.epsilon -= .02
            # agent.epsilon = max(0.05,agent.epsilon)
            print('agent epsilon:', agent.epsilon)

def check():
    do_episode_collect_trajectory(agent,env,max_steps=1000,render=True,feed=False,realtime=True,use_best=True)
