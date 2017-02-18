from __future__ import print_function

# Deep Deterministic Policy Gradient Method
# David Silver et al.

# implemented in plain Keras, by Qin Yongliang
# 2017 01 13

# heavily optimized for speed, lots of numpy flowed into tensorflow
# 2017 01 14

# changed to canton library implementation, much simpler code
# 2017 02 18

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

from math import *
import random
import time
from winfrey import wavegraph

from rpm import rpm # replay memory implementation

from noise import one_fsq_noise

import tensorflow as tf
import canton as ct
from canton import *

class ResDense(Can): # residual dense unit
    def __init__(self,nip):
        super().__init__()
        nbp = int(nip/4)
        d0 = Dense(nip,nbp)
        d1 = Dense(nbp,nip)
        self.d = [d0,d1]
        self.incan(self.d)

    def __call__(self,i):
        inp = i
        i = self.d[0](i)
        i = Act('tanh')(i)
        i = self.d[1](i)
        i = Act('tanh')(i)
        return inp + i

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0)

class nnagent(object):
    def __init__(self,
    observation_space,
    action_space,
    stack_factor=1,
    discount_factor=.99, # gamma
    train_skip_every=1,
    ):
        self.rpm = rpm(1000000) # 1M history
        self.render = True
        self.noise_source = one_fsq_noise()
        self.train_counter = 0
        self.train_skip_every = train_skip_every
        self.observation_stack_factor = stack_factor

        self.inputdims = observation_space.shape[0] * self.observation_stack_factor
        # assume observation_space is continuous

        self.is_continuous = True if isinstance(action_space,Box) else False

        if self.is_continuous: # if action space is continuous

            low = action_space.low
            high = action_space.high

            num_of_actions = action_space.shape[0]

            self.action_bias = high/2. + low/2.
            self.action_multiplier = high - self.action_bias

            # say high,low -> [2,7], then bias -> 4.5
            # mult = 2.5. then [-1,1] multiplies 2.5 + bias 4.5 -> [2,7]

            def clamper(actions):
                return np.clip(actions,a_max=action_space.high,a_min=action_space.low)

            self.clamper = clamper
        else:
            num_of_actions = action_space.n

            self.action_bias = .5
            self.action_multiplier = .5 # map (-1,1) into (0,1)

            def clamper(actions):
                return np.clip(actions,a_max=1.,a_min=0.)

            self.clamper = clamper

        self.outputdims = num_of_actions
        self.discount_factor = discount_factor
        ids,ods = self.inputdims,self.outputdims
        print('inputdims:{}, outputdims:{}'.format(ids,ods))

        self.actor = self.create_actor_network(ids,ods)
        self.critic = self.create_critic_network(ids,ods)
        self.actor_target = self.create_actor_network(ids,ods)
        self.critic_target = self.create_critic_network(ids,ods)

        # print(self.actor.get_weights())
        # print(self.critic.get_weights())

        self.feed,self.joint_inference,sync_target = self.train_step_gen()

        sess = ct.get_session()
        sess.run(tf.global_variables_initializer())

        sync_target()

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        c = Can()
        c.add(Dense(inputdims,64))
        c.add(ResDense(64))
        c.add(ResDense(64))
        c.add(Dense(64,outputdims))

        if self.is_continuous:
            c.add(Act('tanh'))
            c.add(Lambda(lambda x: x*self.action_multiplier + self.action_bias))
        else:
            c.add(Act('softmax'))

        c.chain()
        return c

    # q = critic(s,a) : predict q given state and action
    def create_critic_network(self,inputdims,actiondims):
        c = Can()
        c.add(Lambda(lambda x:tf.concat([x[0],x[1]],axis=1)))
        # concat state and action
        c.add(Dense(inputdims+actiondims,64))
        c.add(ResDense(64))
        c.add(ResDense(64))
        c.add(Dense(64,1))
        c.chain()
        return c

    def train_step_gen(self):
        s1 = tf.placeholder(tf.float32,shape=[None,self.inputdims])
        a1 = tf.placeholder(tf.float32,shape=[None,self.outputdims])
        r1 = tf.placeholder(tf.float32,shape=[None,1])
        isdone = tf.placeholder(tf.float32,shape=[None,1])
        s2 = tf.placeholder(tf.float32,shape=[None,self.inputdims])

        # 1. update the critic
        a2 = self.actor_target(s2)
        q2 = self.critic_target([s2,a2])
        q1_target = r1 + (1-isdone) * self.discount_factor * q2
        q1_predict = self.critic([s1,a1])
        critic_loss = tf.reduce_mean((q1_target - q1_predict)**2)
        # produce better prediction

        # 2. update the actor
        a1_predict = self.actor(s1)
        q1_predict = self.critic([s1,a1_predict])
        actor_loss = tf.reduce_mean(- q1_predict)
        # maximize q1_predict -> better actor

        # 3. shift the weights
        tau = tf.Variable(0.001)
        aw = self.actor.get_weights()
        atw = self.actor_target.get_weights()
        cw = self.critic.get_weights()
        ctw = self.critic_target.get_weights()

        shift1 = [tf.assign(atw[i], aw[i]*tau + atw[i]*(1-tau))
            for i,_ in enumerate(aw)]
        shift2 = [tf.assign(ctw[i], cw[i]*tau + ctw[i]*(1-tau))
            for i,_ in enumerate(cw)]

        # 4. inference
        a_infer = self.actor(s1)
        q_infer = self.critic([s1,a_infer])
        # actions = actor.infer(obs)
        # q = critic.infer([obs,actions])[0]

        # optimizer on
        # opt = tf.train.MomentumOptimizer(1e-1,momentum=0.9)
        opt = tf.train.RMSPropOptimizer(1e-4)
        cstep = opt.minimize(critic_loss,
            var_list=self.critic.get_weights())
        astep = opt.minimize(actor_loss,
            var_list=self.actor.get_weights())

        def feed(memory):
            [s1d,a1d,r1d,isdoned,s2d] = memory # d prefix means data
            sess = ct.get_session()
            res = sess.run([critic_loss,actor_loss,
                cstep,astep,shift1,shift2],
                feed_dict={
                s1:s1d,a1:a1d,r1:r1d,isdone:isdoned,s2:s2d,tau:1e-3
                })
            # print('closs: {:6.4f} aloss: {:6.4f}'.format(
            # res[0],res[1]))

        def joint_inference(state):
            sess = ct.get_session()
            res = sess.run([a_infer,q_infer],feed_dict={s1:state})
            return res

        def sync_target():
            sess = ct.get_session()
            sess.run([shift1,shift2],feed_dict={tau:1.})

        return feed,joint_inference,sync_target

    def train(self,verbose=1):
        memory = self.rpm
        batch_size = 64
        total_size = batch_size * self.train_skip_every
        epochs = 1

        self.train_counter+=1
        self.train_counter %= self.train_skip_every

        if self.train_counter != 0: # train every few steps
            return

        if memory.size() > total_size:
            #if enough samples in memory

            # sample randomly a minibatch from memory
            [s1,a1,r1,isdone,s2] = memory.sample_batch(total_size)
            # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)

            self.feed([s1,a1,r1,isdone,s2])

    def feed_one(self,tup):
        self.rpm.add(tup)

    # gymnastics
    def play(self,env,max_steps=-1,realtime=False,noise_level=0.): # play 1 episode
        timer = time.time()
        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0

        # stack a little history to ensure markov property
        # LSTM will definitely be used here in the future...
        # global que # python 2 quirk
        self.que = np.zeros((self.inputdims,),dtype='float32') # list of recent history actions

        def quein(observation):
            # global que # python 2 quirk
            length = len(observation)
            self.que[0:-length] = self.que[length:] # left shift
            self.que[-length:] = np.array(observation)

        def quecopy():
            return self.que.copy()

        # what the agent see as state is a stack of history observations.

        observation = env.reset()
        quein(observation) # quein o1

        while True and steps <= max_steps:
            steps +=1

            thisque = quecopy() # s1

            action = self.act(thisque) # a1

            if self.is_continuous:
                # add noise to our actions, since our policy by nature is deterministic
                exploration_noise = self.noise_source.one((self.outputdims,),noise_level)
                exploration_noise *= self.action_multiplier
                # print(exploration_noise,exploration_noise.shape)
                action += exploration_noise
                action = self.clamper(action)
                action_out = action
            else:
                exploration_noise = self.noise_source.one((self.outputdims,),noise_level)
                exploration_noise *= self.action_multiplier
                action += exploration_noise
                # action = self.clamper(action)
                action = softmax(action)
                # discretize our actions
                probabilities = action
                csprob = np.cumsum(probabilities)
                action_index = (csprob > np.random.rand()).argmax()
                action_out = action_index

            # o2, r1,
            observation, reward, done, _info = env.step(action_out)

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            quein(observation) # quein o2
            nextque = quecopy() # s2

            # feed into replay memory
            self.feed_one((thisque,action,reward,isdone,nextque)) # s1,a1,r1,isdone,s2

            if self.render==True and (steps%10==0 or realtime==True):
                env.render()
            if done :
                break

            verbose= 2 if steps==1 else 0
            self.train(verbose=verbose)

        # print('episode done in',steps,'steps',time.time()-timer,'second total reward',total_reward)
        totaltime = time.time()-timer
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward
        ))
        return

    # one step of action, given observation
    def act(self,observation):
        actor,critic = self.actor,self.critic
        obs = np.reshape(observation,(1,len(observation)))

        # actions = actor.infer(obs)
        # q = critic.infer([obs,actions])[0]
        [actions,q] = self.joint_inference(obs)
        q = q[0]

        disp_actions = (actions[0]-self.action_bias) / self.action_multiplier
        disp_actions = disp_actions * 5 + np.arange(self.outputdims) * 12.0 + 30

        noise = self.noise_source.ask() * 5 - np.arange(self.outputdims) * 12.0 - 30

        self.loggraph(np.hstack([disp_actions,noise,q]))

        return actions[0]

    def loggraph(self,waves):
        if not hasattr(self,'wavegraph'):
            def rn():
                r = np.random.uniform()
                return 0.2+r*0.4
            colors = []
            for i in range(len(waves)-1):
                color = [rn(),rn(),rn()]
                colors.append(color)
            colors.append([0.2,0.5,0.9])
            self.wavegraph = wavegraph(len(waves),'actions/noises/Q',np.array(colors))

        wg = self.wavegraph
        wg.one(waves.reshape((-1,)))

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

# p = playground('LunarLanderContinuous-v2')
p = playground('Pendulum-v0')
# p = playground('MountainCar-v0')BipedalWalker-v2
# p = playground('BipedalWalker-v2')

e = p.env

agent = nnagent(
e.observation_space,
e.action_space,
discount_factor=.99,
stack_factor=1,
train_skip_every=1,
)

def r(ep):
    # agent.render = True
    e = p.env
    noise_level = .05
    for i in range(ep):
        noise_level *= .95
        noise_level = max(1e-11,noise_level - 1e-4)
        print('ep',i,'/',ep,'noise_level',noise_level)
        agent.play(e,realtime=True,max_steps=-1,noise_level=noise_level)

def test():
    e = p.env
    agent.render = True
    agent.play(e,realtime=True,max_steps=-1,noise_level=1e-11)
