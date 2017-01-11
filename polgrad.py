from __future__ import print_function
import numpy as np
import gym
from gym.spaces import Discrete, Box

# (naive) policy gradient method for Reinforcement Learning
# by Qin Yongliang
# 2017 01 06

'''
for English readers: this is the only section written in Chinese. Just skip it, everything important is written in English(scroll down to see.)

To understand this piece of code (even with comments), you should be familiar with the training of supervised neural networks.

写在前面：各位读者，这份代码实现的是增强学习的“决策梯度”方法。
要读懂这个代码（虽然很多注释），你至少应该熟悉怎样训练监督神经网络。

假设我们有一个神经网络，输入是当前的环境状态，输出是下一步可选的每一个操作的概率。这里的输出一般称为【决策】。
初始状态下，我们应该会对所有操作输出一样的概率，也就是随机决策。

要训练神经网络，我们就必须提供【每一种给定状态下，能够最大化未来收益的操作】。
然后我们就可以用梯度下降法训练神经网络，提高网络输出这些操作的概率。可是我们没有这个数据。

增强学习要解决的问题，正是每做一步操作时，并不都会获得奖励（所以没办法确定每一步是好是坏）。但我们还是可以用一个函数，近似地评估之前每一步操作的“牛逼程度”，这样我们仍然可以利用梯度下降对网络进行训练，以获得更好的决策。

由于一个神经网络的行为，也就是【决策】，由这个网络的参数决定，而我们需要将这些参数对预期收益的梯度，进行梯度下降（或者上升）来最大化收益，因此这个方法称为“决策梯度”方法。

在完成一个回合(one episode)之后，如果我们可以给我们的每一步行为(actions)打一个“牛逼程度”(coolness)的分数，我们就可以训练一个神经网络模型，输入是每一步看到的环境状态(state)，输出是给定状态下选择每一个行为的概率(probability of each action)，误差函数是 【- mean(log(网络输出的概率分布)*行为的牛逼程度)】。这样，我们就可以将网络的输出（也就是决策的概率分布）中，具有更高期望收益的那些操作的发生概率提高，反之亦然。

很多教程里用的说法是梯度上升，我在mean前面加了负号，转换成梯度下降，于是就可以用keras这样的框架进行训练，极大地减少了自己写训练代码的痛苦。

给每个操作打牛逼程度(coolness)分数的原理和具体方法，在下面用英文给出，代码非常简单。

我用的是最简单的【时间指数折扣】(time exponential reward)，根据每一个操作与获得的某个奖励之间的时间间隔给该操作打分（操作和奖励之间隔得越久，该操作对最终获得奖励的贡献就越少，该操作得分越低），这个方法的bias低，但是variance高，收敛较慢。

'''

# evaluate the coolness of a list of actions, based on the rewards.
def evaluate_coolness(rewards,discount_factor=.99):
    # you performed one episode, collected a list of actions and rewards.
    length = len(rewards)

    """
    actions[i] -> what you did in step i
    rewards[i] -> what you got in return after action i

    coolness[i] -> how cool is action i

    for each action in list we evaluate its coolness.

    first, set all coolnesses to zero:
    """

    coolness = np.zeros((length,1))

    """
    intuitively,
    an action's coolness is the sum of rewards after you performed that action.
    coolness[i] = rewards[i] + rewards[i+1] + rewards[i+2]...

    but in fact, the relationship between actions and
    rewards decreases as the time between them increases.

    that is because the world is an stochastic process.
    your actions can only affect your NEAR future.
    statistically, the relationship decreases exponentially.
    that's why you can predict next minute but can't predict next month.

    formally speaking,
    P(reward|action) = 0.9 ^ time_between_them

    where gamma = 0.9 is the DISCOUNT FACTOR.

    therefore,
    coolness[i] = rewards[i] + rewards[i+1] * 0.9 + rewards[i+2] * 0.9 * 0.9...
    as stated above, the coolness decreases exponentially for future rewards.

    or equivalently,
    for each reward i, we increase the coolness of actions 0 to i-1.
    coolness[i] += rewards[i]
    coolness[i-1] += rewards[i] * 0.9
    coolness[i-2] += rewards[i] * 0.9 * 0.9
    ...

    """

    if discount_factor<1.0 : # if discount_factor comes into play
        # for each reward:
        for i in range(length):
            reward_i = rewards[i]

            # what we meant to do:
            # coolness[i] += reward[i]
            # coolness[i-1] += reward[i] * .9
            # coolness[i-2] += reward[i] * .9 * .9
            # coolness[i-3] += reward[i] * .9 * .9 * .9

            # what we actually did (to optimize for speed):
            for k in reversed(range(i+1)): # iterate k from i to 0
                coolness[k] += reward_i
                reward_i *= discount_factor

    else:
        # if you intentionally set discount_factor to 1, which means no discount
        # then the following code will be faster:
        # for each reward:
        for i in range(length):
            reward_i = rewards[i]

            # what we meant to do:
            # coolness[i] += reward
            # coolness[i-1] += reward
            # coolness[i-2] += reward
            # coolness[i-3] += reward

            # what we actually did (to optimize for speed):
            coolness[0:i+1] += reward_i


    # done.
    return coolness

# in order to play episodes with neural network,
# we have to generate training examples for a neural network, given the history of
# observations, actions and rewards.

# ideally,
# network input = what we observe before each step of action
# network output = probability distribution of suggested actions for that step.

# so, to train the network, we have to produce:
# example_x = what we observed before each step of action
# example_y = coolness indicator of each of the actions, calculated from the rewards.

# then we ask the network to maximize coolness.

# here is the coolness indicator:
def multiply_coolness_with_actions(actions,coolness):
    # you performed one episode, collected a list of actions and rewards.
    length = len(actions)

    # actions[i] -> what you did in step i
    # coolness[i] -> how cool is action i

    '''
    now, coolness[i] is actions[i]'s coolness score.

    after evaluating the coolness of each action,
    we could use that coolness information to adjust
    the probability of our actions.

    let's say that:
    - if an action is cooler than others, it should be more probable;
    - if an action is less cool than others, it should be less probable.

    in neural network scenario, we know that the network outputs are
    usually the probability of each actions, given the observations.
    or: probability_of_actions = neural_network(observations)

    then we could choose our actions based on network_output.

    for example, if the network outputs
    [0.3, 0.3, 0.3]
    that means we will randomly choose an action out of 3.

    now let's suppose we chose the first action.
    we can describe that action as [1.0, 0.0, 0.0].

    after one episode, we found out that specific action was actually cool
    (with a high coolness score),
    so we now want the network to output a higher probability on that action.
    or, to raise the value of network_output[0].
    or [0.5, 0.25, 0.25].
    but how?

    what we do: we simply multiply the actions with its coolness.
    for a coolness of 3, we get action_coolness of [3.0, 0.0, 0.0]
    for a coolness of -3, we get action_coolness of [-3.0, 0.0, 0.0]

    then we ask the neural network to:
    maximize(log(network_output) * action_coolness)
    this will result in network try to raise probability for cooler actions.

    for example, if an action is cool, say action_coolness [1,0,0]
    then in order to maximize(log(network_output) * action_coolness),
    the network will have to maximize(log(network_output[0]))
    therefore maximizes(network_output[0]).
    we will then get something like [0.5, 0.25, 0.25].

    otherwise, if an action is uncool, say action_coolness [-1,0,0]
    then in order to maximize(log(network_output) * action_coolness),
    the network will have to minimize(log(network_output[0]))
    therefore minimizes(network_output[0]).
    we will then get something like [0.2, 0.4, 0.4].

    what the log() does:
    if the action rewards are all negative, say [-100,0,0], [-150,0,0], etc.
    then with log(probability), we don't have to decrease
    the probability of the actions to below zero.
    (which is impossible since probability by definition has to be >= 0)

    instead, we utilize the fact that
    log(something_close_to_zero) = very_negative_value
    so, by adding a log() to the probability,
    we could produce negative values, with positive probabilities.

    Now that's what they call POLICY GRADIENT methods.

    let's assume our actions are already one-hot representations:
    (which means choosing action 0 -> [1,0,0] and choosing action 1 -> [0,1,0])
    '''

    action_coolness = actions * coolness

    # done
    return action_coolness

# run episode with some policy of some agent, and collect the rewards
# well the usual gym stuff
def do_episode_collect_trajectory(agent, env, max_steps, render=True, feed=True, realtime=False, use_best=False):
    # keep a record of:
    observations=[] # what we see,
    actions=[] # what we did,
    rewards=[] # what we get, each time.

    observation = env.reset()
    agent.wakeup() # notify the agent

    for t in range(max_steps):
        observations.append(observation)
        action = agent.act(observations[-1],use_best=use_best)
        # acting based on last observation and last observation only.

        actions.append(action)
        (observation, reward, done, _info) = env.step(action)
        rewards.append(reward)

        if render and (t%10==0 or realtime==True): env.render()
        if done : break


    if feed:
        agent.feed_episodic_data((observations,actions,rewards))
    return observations, actions, rewards

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
        self.state_coolness = np.zeros((0,1))
        self.action_coolness = np.zeros((0,num_of_actions))
        self.actions = np.zeros((0,num_of_actions))

        self.num_of_actions = num_of_actions
        self.num_of_observations = num_of_observations
        self.discount_factor = discount_factor

        self.epsilon = epsilon # epsilon-greedy per David Silver's Lecture.

        # our network
        input_shape = num_of_observations
        i = Input(shape=(input_shape,))
        h = Dense(20,activation='tanh')(i)

        # h = Dense(num_of_actions,activation='tanh')(h)

        h = Dense(num_of_actions)(h)
        out = Activation('softmax')(h)

        model = Model(input=i,output=out)

        # as described in section above, we want to
        # maximize(log(network_output) * action_coolness)

        # or, minimize the negative of it:
        def policy_loss(y_true,y_pred):
            network_output = y_pred
            action_coolness = y_true
            return - K.mean(K.log(network_output+1e-10) * action_coolness)

        model.compile(loss=policy_loss,optimizer=optimizer)
        self.model = model
        print('agent Initialized with',num_of_observations,'dim input and',num_of_actions,'dim output.')
        print('discount_factor',discount_factor)
        print('model architechture:')
        model.summary()

    # act one step base on observation
    def act(self, observation, use_best=False):
        model = self.model
        epsilon = self.epsilon # greedy factor

        observation = observation.reshape((1,len(observation)))

        # observation is a vector
        probabilities = model.predict([observation])[0]

        # here are two ways to draw actions considering their probabilities

        if self.epsilon < 0: # if epsilon is set to impossible values
            csprob = np.cumsum(probabilities)
            # 'cumsum' turns [0.3, 0.5, 0.1, 0.1] into [0.0, 0.3, 0.8, 0.9]
            # so that we could draw from it using a random number from 0 - 1

            # draw one action, with probability of that action.
            # then return the index of that action (an integer)
            action_index = (csprob > np.random.rand()).argmax()

        if self.epsilon > 0:
            # with probability epsilon we act greedy:
            if self.epsilon > np.random.rand(1):
                action_index = probabilities.argmax()
            else:
                # with probability 1-epsilon we act randomly:
                action_index = np.random.choice(len(probabilities))

        if use_best:
            # always act greedly
            action_index = probabilities.argmax()

        return action_index

    def wakeup(self):
        # clear states
        pass

    # after playing for one(or whatever) episode, we could feed the agent with data.
    def feed_episodic_data(self,episodic_data):
        observations,actions,rewards = episodic_data

        actions = np.array(actions)

        # IMPORTANT: convert actions to their one-hot representations
        def one_hot(tensor,classes):
            heat = np.zeros(tensor.shape+(classes,))
            for i in range(classes):
                heat[...,i] = tensor[...] == i
            return heat
        onehot_actions = one_hot(actions,self.num_of_actions)

        # how cool is each action?
        coolness = evaluate_coolness(rewards,self.discount_factor)
        action_coolness = multiply_coolness_with_actions(onehot_actions,coolness)

        # how cool is each state?
        # state_coolness = evaluate_coolness(rewards,discount_factor=1.)

        # add to agent's database
        self.observations = np.vstack((self.observations,observations))
        self.action_coolness = np.vstack((self.action_coolness,action_coolness))

    # train agent with some of its collected data from its database
    def train(self,epochs=100):
        model = self.model
        observations,coolness = self.observations, self.action_coolness
        length = len(observations)


        for i in range(epochs):
            # train 1 epoch on a randomly selected subset of the whole database.

            subset_size = min(length,256)

            indices = np.random.choice(length,subset_size,replace=False)
            subset_observations = np.take(observations,indices,axis=0)
            subset_coolness = np.take(coolness,indices,axis=0)

            model.fit(subset_observations, subset_coolness,
                      batch_size=len(subset_coolness),
                      nb_epoch=1,
                      shuffle=False)

    # (optionally) throw away all data
    def dump(self):
        pass
        self.observations = np.zeros((0,self.num_of_observations))
        self.action_coolness = np.zeros((0,self.num_of_actions))

    def bad_kids_eaten_by_the_wolf(self): # discard the least cool actions from database.
        num_stay = 50000
        length = len(self.action_coolness)
        if length <= num_stay:
            return
        else:
            print('wolf ate',length - num_stay,'kids')

        coolness = np.mean(self.action_coolness,axis=-1)
        stay_indices = np.argsort(coolness)[length-num_stay:length]

        self.action_coolness = np.array([self.action_coolness[i] for i in stay_indices])
        self.observations = np.array([self.observations[i] for i in stay_indices])

    def drop_previous(self): # optional: discard early samples. not very useful for most cases
        num_stay = 3000
        length = len(self.action_coolness)
        if length <= num_stay:
            return
        else:
            print('wolf ate',length - num_stay,'kids')

        self.action_coolness = self.action_coolness[length - num_stay:length]
        self.observations = self.observations[length - num_stay:length]

from gym import wrappers

# give it a try
# env = gym.make('Acrobot-v1')
env = gym.make('CartPole-v1')
# env = wrappers.Monitor(env,'./experiment-3',force=True)
agent = nnagent(
num_of_actions=env.action_space.n,
num_of_observations=env.observation_space.shape[0],
discount_factor=1.,
epsilon=-1,
optimizer = SGD(lr=0.003)
)

# main training loop
def r(times=3):
    for k in range(times):
        print('training loop',k,'/',times)
        for i in range(1): # do 1 episode
            print('play episode:',i)
            episodic_data = do_episode_collect_trajectory(agent,env,max_steps=5000,render=True,feed=True)
            # after play, the episodic data will be feeded to the agent AUTOMATICALLY, so no feeding here

            print('length of episode:',len(episodic_data[0]))
            print('total reward of episode:',np.sum(episodic_data[2]))

        if len(agent.observations)> 300: # wait until collected data became diverse enough
            # ask agent to train itself, with previously collected data
            agent.train(epochs=min(500,len(agent.observations)))

            # increase epsilon to make agent choose less and less random actions.
            # if agent.epsilon>0:
            #     agent.epsilon+=0.005
            #     print('agent epsilon:', agent.epsilon)

def check():
    episodic_data = do_episode_collect_trajectory(agent,env,max_steps=1000,render=True,feed=False,realtime=True,use_best=True)
    print('length of episode:',len(episodic_data[0]))
    print('total reward of episode:',np.sum(episodic_data[2]))
