from __future__ import print_function

# Deep Deterministic Policy Gradient Method
# David Silver et al.

# implemented in plain Keras, by Qin Yongliang
# 2017 01 13

# heavily optimized for speed, lots of numpy flowed into tensorflow
# 2017 01 14

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
from winfrey import wavegraph

from rpm import rpm # replay memory implementation

from noise import one_fsq_noise

def bn(i):
    return i
    return BatchNormalization(mode=1)(i)

def relu(i):
    return Activation('relu')(i)

# residual dense unit
def resdense(idim,odim):
    def unit(i):
        mdim = max(4,int(idim/4),int(odim/4))

        if idim==odim:
            ident = i
            i = bn(i)
            i = relu(i)
            i = Dense(mdim)(i)

            i = bn(i)
            i = relu(i)
            i = Dense(odim)(i)

        else:
            i = bn(i)
            i = relu(i)
            ident = i
            i = Dense(mdim)(i)

            i = bn(i)
            i = relu(i)
            i = Dense(odim)(i)

            ident = Dense(odim)(ident)

        out = merge([ident,i],mode='sum')
        return out
    return unit

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
    optimizer=RMSprop()
    ):
        self.rpm = rpm(1000000) # 1M history
        self.noise_source = one_fsq_noise()
        self.train_counter = 0
        self.train_skip_every = 1
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
        self.optimizer = optimizer

        ids,ods = self.inputdims,self.outputdims
        self.actor, self.frozen_actor = self.create_actor_network(ids,ods)
        self.critic, self.frozen_critic = self.create_critic_network(ids,ods)

        print('inputdims:{}, outputdims:{}'.format(ids,ods))
        print('actor network:')
        self.actor.summary()
        print('critic network:')
        self.critic.summary()

        # target networks: identical copies of actor and critic
        self.actor_target,self.frozen_actor_target = self.create_actor_network(ids,ods)
        self.critic_target, self.frozen_critic_target = self.create_critic_network(ids,ods)

        self.replace_weights(tau=1.)

        self.create_q1_target_model()
        self.create_actor_trainer()
        self.create_critic_trainer()

    def create_actor_trainer(self):
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
    def _replace_weights(self,tau=0.001):
        theta_a,theta_c = self.actor.get_weights(),self.critic.get_weights()
        theta_a_targ,theta_c_targ = self.actor_target.get_weights(),self.critic_target.get_weights()

        # mixing factor tau : we gradually shift the weights...
        theta_a_targ = [theta_a[i]*tau + theta_a_targ[i]*(1-tau) for i in range(len(theta_a))]
        theta_c_targ = [theta_c[i]*tau + theta_c_targ[i]*(1-tau) for i in range(len(theta_c))]

        self.actor_target.set_weights(theta_a_targ)
        self.critic_target.set_weights(theta_c_targ)

    # the method above uses numpy, how can we flow it in tensorflow?
    def replace_weights(self,tau=0.001):
        if not hasattr(self,'wflow'):
            self.wflow = self.weights_flow()

        flow = self.wflow
        tau = np.array([tau],dtype='float32')

        flow([tau,0])

    def weights_flow(self):
        # define the weight replacing op
        theta_a,theta_c = self.actor.weights,self.critic.weights
        theta_a_targ,theta_c_targ = self.actor_target.weights,self.critic_target.weights

        tau_place = K.placeholder(shape=(1,))

        ops = []
        for i,w in enumerate(theta_a_targ):
            ops += [theta_a_targ[i].assign(theta_a[i]*tau_place + theta_a_targ[i]*(1-tau_place))]

        for i,w in enumerate(theta_c_targ):
            ops += [theta_c_targ[i].assign(theta_c[i]*tau_place + theta_c_targ[i]*(1-tau_place))]

        flow = K.function([tau_place],ops)
        return flow

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        inp = Input(shape=(inputdims,))
        i = inp
        i = Dense(128)(i)
        i = resdense(128,128)(i)
        # i = resdense(128,128)(i)
        i = relu(bn(i))

        i = Dense(outputdims)(i)

        if self.is_continuous:
            # map into (-1,1)
            i = Activation('tanh')(i)
            # map into action_space
            i = Lambda(lambda x:x * self.action_multiplier + self.action_bias)(i)
        else:
            # map into (0,1)
            i = Activation('softmax')(i)

        out = i
        model = Model(input=inp,output=out)

        # now we create a frozen_model,
        # that uses the same layers with weights frozen when trained.
        frozen_model = Model(input=inp,output=out)
        frozen_model.trainable = False

        return model,frozen_model

    # q = critic(s,a) : predict q given state and action
    def create_critic_network(self,inputdims,actiondims):
        inp = Input(shape=(inputdims,))
        act = Input(shape=(actiondims,))
        i = merge([inp,act],mode='concat')

        i = Dense(128)(i)
        i = resdense(128,128)(i)
        # i = resdense(128,128)(i)
        i = relu(bn(i))

        i = Dense(1)(i)
        out = i
        model = Model(input=[inp,act],output=out)

        # now we create a frozen_model,
        # that uses the same layers with weights frozen when trained.
        frozen_model = Model(input=[inp,act],output=out)
        frozen_model.trainable = False

        return model,frozen_model

    def create_q1_target_model(self):
        # this part is for performance optimization
        # for explaination of this part, please check train()

        s2i = Input(shape=(self.inputdims,))
        a2i = self.frozen_actor_target(s2i)
        q2i = self.frozen_critic_target([s2i,a2i])

        r1i = Input(shape=(1,))
        isdonei = Input(shape=(1,))

        def calc_q1_target(x):
            [r1i,isdonei,q2i] = x
            return r1i + (1-isdonei) * self.discount_factor * q2i

        def calc_output_shape(input_shapes):
            return input_shapes[0]

        q1_target = merge([r1i,isdonei,q2i],mode=calc_q1_target,output_shape=calc_output_shape)
        q1_target_model = Model(input=[s2i,r1i,isdonei],output=q1_target)

        self.q1_target_model = q1_target_model

    def create_critic_trainer(self):
        # this part is also for performance optimization...

        qtm = self.q1_target_model
        qtm.trainable = False

        s1i = Input(shape=(self.inputdims,))
        s2i = Input(shape=(self.inputdims,))
        a1i = Input(shape=(self.outputdims,))

        r1i = Input(shape=(1,))
        isdonei = Input(shape=(1,))

        q1t = qtm([s2i,r1i,isdonei])
        crit = self.critic([s1i,a1i])

        def mse(x):
            return (x[0]-x[1])**2

        def calc_output_shape(input_shapes):
            return input_shapes[0] # shape of r1i

        loss = merge([q1t,crit],mode=mse,output_shape=calc_output_shape)

        def thru(y_true,y_pred):
            return y_pred

        model = Model(input=[s1i,a1i,r1i,isdonei,s2i],output=loss)
        model.compile(loss=thru,optimizer=self.optimizer)

        self.critic_trainer = model

    def train(self,verbose=1):
        memory = self.rpm
        critic,frozen_critic = self.critic,self.frozen_critic
        actor = self.actor
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

            if False: # the following is optimized away but kept for clarity.

                # a2_targ = actor_targ(s2) : what will you do in s2, Mr. old actor?
                a2 = self.actor_target.predict(s2)

                # q2_targ = critic_targ(s2,a2) : how good is action a2, Mr. old critic?
                q2 = self.critic_target.predict([s2,a2])

                # what if we combine the 2 above to improve performance?
                s2i = Input(shape=(self.inputdims,))
                a2i = self.actor_target(s2i)
                q2i = self.critic_target([s2i,a2i])

                # if a2 is q2-good, then what should q1 be?
                # Use Bellman Equation! (recursive definition of q-values)
                # if not last step of episode:
                #   q1 = (r1 + gamma * q2)
                # else:
                #   q1 = r1

                q1_target = r1 + (1-isdone) * self.discount_factor * q2

                # but, what if we combine all above to improve performance?
                r1i = Input(shape=(1,))
                isdonei = Input(shape=(1,))

                def calc_q1_target(x):
                    [r1i,isdonei,q2i] = x
                    return r1i + (1-isdonei) * self.discount_factor * q2i

                def calc_output_shape(input_shapes):
                    return input_shapes[0]

                q1_target = merge([r1i,isdonei,q2i],mode=calc_q1_target,output_shape=calc_output_shape)
                q1_target_model = Model(input=[s2i,r1i,isdonei],output=q1_target)

            else:

                # q1_target_model is already implemented in create_q1_target_model()
                # q1_target = self.q1_target_model.predict([s2,r1,isdone])

                # all above were optimized away...
                critic_trainer = self.critic_trainer

            # critic.fit([s1,a1],
            # q1_target,
            # batch_size=batch_size,
            # nb_epoch=epochs,
            # verbose=verbose,
            # shuffle=False
            # )

            critic_trainer.fit([s1,a1,r1,isdone,s2],
            np.zeros((total_size,1)), # useless target label
            batch_size=batch_size,
            nb_epoch=epochs,
            verbose=verbose,
            shuffle=False
            )

            # now the critic can predict more accurate q given s and a.
            # thanks to the Bellman equation, and David Silver.

            # with a better critic, we can now improve our actor!

            if False: # the following part is optimized away. left here for explaination purposes

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

            else: # the actor_trainer is already initialized in create_actor_trainer()
                actor_trainer = self.actor_trainer

                actor_trainer.fit(s1,
                np.zeros((total_size,1)), # useless target label
                batch_size=batch_size,
                nb_epoch=epochs,
                verbose=verbose,
                shuffle=False
                )

            # now both the actor and the critic have improved.
            self.replace_weights(tau=0.001 * self.train_skip_every)

        else:
            pass
            # print('# no enough samples, not training')

    def feed_one(self,tup):
        self.rpm.add(tup)

    # gymnastics
    def play(self,env,max_steps=-1,realtime=False,render=True,noise_level=0.): # play 1 episode
        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0

        # stack a little history to ensure markov property
        # LSTM will definitely be used here in the future...
        global que # python 2 quirk
        que = np.zeros((self.inputdims,),dtype='float32') # list of recent history actions

        def quein(observation):
            global que # python 2 quirk
            length = len(observation)
            que[0:-length] = que[length:] # left shift
            que[-length:] = np.array(observation)

        # what the agent see as state is a stack of history observations.

        observation = env.reset()
        quein(observation) # quein o1

        while True and steps <= max_steps:
            steps +=1

            lastque = que.copy() # s1

            action = self.act(lastque) # a1

            if self.is_continuous:
                # add noise to our actions, since our policy by nature is deterministic
                exploration_noise = self.noise_source.one((self.outputdims,),noise_level)
                exploration_noise *= self.action_multiplier
                # print(exploration_noise,exploration_noise.shape)
                action += exploration_noise
                action = self.clamper(action)
                action_out = action
            else:
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
            nextque = que.copy() # s2

            # feed into replay memory
            self.feed_one((lastque,action,reward,isdone,nextque)) # s1,a1,r1,isdone,s2

            if render and (steps%10==0 or realtime==True): env.render()
            if done :
                break

            verbose= 2 if steps==1 else 0
            self.train(verbose=verbose)

        print('episode done in',steps,'steps, total reward',total_reward)
        return

    # one step of action, given observation
    def act(self,observation):
        actor,critic = self.actor,self.critic
        obs = np.reshape(observation,(1,len(observation)))
        actions = actor.predict(obs)

        q = critic.predict([obs,actions])[0]

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
# p = playground('Pendulum-v0')
# p = playground('MountainCar-v0')BipedalWalker-v2
p = playground('BipedalWalker-v2')

e = p.env

agent = nnagent(
e.observation_space,
e.action_space,
discount_factor=.996,
optimizer=RMSprop()
discount_factor=.99,
stack_factor=1,
optimizer='rmsprop'
)

def r(ep):
    e = p.env
    noise_level = 1.
    for i in range(ep):
        noise_level *= .98
        noise_level = max(1e-4,noise_level)
        print('ep',i,'/',ep,'noise_level',noise_level)
        agent.play(e,max_steps=-1,noise_level=noise_level)
