from __future__ import print_function
import numpy as np
import gym
from gym.spaces import Discrete, Box

# policy gradient method
# by Qin Yongliang
# 2017 01 06

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

    if False:
        # for each reward:
        for i in range(length):
            reward_i = rewards[i]

            # what we meant to do:
            # coolness[i] += reward
            # coolness[i-1] += reward * .9
            # coolness[i-2] += reward * .9 * .9
            # coolness[i-3] += reward * .9 * .9 * .9

            # what we actually did (to optimize for speed):
            for k in reversed(range(i+1)): # iterate k from i to 0
                coolness[k] += reward_i
                # reward_i *= .9
                reward_i *= discount_factor
    else:
        # for each reward:
        for i in range(length):
            reward_i = rewards[i]

            # what we meant to do:
            # coolness[i] += reward
            # coolness[i-1] += reward * .9
            # coolness[i-2] += reward * .9 * .9
            # coolness[i-3] += reward * .9 * .9 * .9

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
# example_y = coolness indicatior of each of the actions, given the rewards

# then ask the network to maximize coolness.

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

    now let's suppose the network chose the first action.
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

    Now that's what I call POLICY GRADIENT methods.

    a.k.a. Fuck some of the YouTube tutorials.

    let's assume our actions are already one-hot representations:
    (which means choosing action 0 -> [1,0,0] and choosing action 1 -> [0,1,0])
    '''

    action_coolness = actions * coolness

    # done
    return action_coolness

# run episode with some policy of some agent, and collect the rewards

def do_episode_collect_trajectory(agent, env, max_steps, render=True, feed=True, realtime=False):
    # keep a record of:
    observations=[] # what we see,
    actions=[] # what we did,
    rewards=[] # what we get, each time.

    observation = env.reset()
    agent.wakeup() # notify the agent

    for t in range(max_steps):
        observations.append(observation)
        action = agent.act(observations[-1])
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
    def __init__(self, num_of_actions, num_of_observations, discount_factor):
        # agent database: only those two are important
        self.observations = np.zeros((0,num_of_observations))
        self.action_coolness = np.zeros((0,num_of_actions))
        self.actions = np.zeros((0,num_of_actions))

        self.num_of_actions = num_of_actions
        self.num_of_observations = num_of_observations
        self.discount_factor = discount_factor

        # our network
        input_shape = num_of_observations
        i = Input(shape=(input_shape,))
        h = Dense(10,activation='tanh')(i)
        # h = Dense(2,activation='tanh')(i)

        # h = Dense(4,activation='tanh')(h)
        # h = Dense(4,activation='tanh')(h)
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

        model.compile(loss=policy_loss,optimizer=Adam())
        self.model = model
        print('agent Initialized with',num_of_observations,'dim input and',num_of_actions,'dim output.')
        print('discount_factor',discount_factor)
        print('model architechture:')
        model.summary()

    # act base on observation
    def act(self, observation):
        model = self.model
        observation = observation.reshape((1,len(observation)))

        # assume observation is a vector
        probabilities = model.predict([observation])[0]

        csprob = np.cumsum(probabilities)
        # turn [.3, .5, .2] into [.3, .8, 1.]

        # pick one action with probability of that action.
        # return the index of that action (integer)
        action_index = (csprob > np.random.rand()).argmax()

        return action_index

    def wakeup(self):
        # clear states
        pass

    # after playing for one episode, we feed the agent with data.
    def feed_episodic_data(self,episodic_data):
        observations,actions,rewards = episodic_data

        actions = np.array(actions)

        # IMPORTANT: convert actions to one-hot
        def one_hot(tensor,classes):
            heat = np.zeros(tensor.shape+(classes,))
            for i in range(classes):
                heat[...,i] = tensor[...] == i
            return heat
        onehot_actions = one_hot(actions,self.num_of_actions)

        # how cool is each action?
        coolness = evaluate_coolness(rewards,self.discount_factor)

        action_coolness = multiply_coolness_with_actions(onehot_actions,coolness)

        # add to agent's database
        self.observations = np.vstack((self.observations,observations))
        self.action_coolness = np.vstack((self.action_coolness,action_coolness))

    # train agent with ALL collected data
    def train(self,epochs=100):
        model = self.model
        observations,coolness = self.observations, self.action_coolness

        model.fit(observations, coolness,
                  batch_size=min(len(observations),5000),
                  nb_epoch=epochs,
                  shuffle=False)

    # (optionally) throw away all data
    def dump(self):
        pass
        self.observations = np.zeros((0,self.num_of_observations))
        self.action_coolness = np.zeros((0,self.num_of_actions))

    def bad_kids_eaten_by_the_wolf(self): # discard the least cool actions from history.
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

    def drop_previous(self):
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
discount_factor=.999999
)

def r(times=3):

    for k in range(times):
        print('training loop',k,'/',times)
        for i in range(1): # do 1 episode
            print('play episode:',i)
            episodic_data = do_episode_collect_trajectory(agent,env,max_steps=5000,render=True,feed=True)
            print('length of episode:',len(episodic_data[0]))
            print('total reward of episode:',np.sum(episodic_data[2]))

        if len(agent.observations)> 200: # wait until data became diverse enough
            agent.train(epochs=min(140,len(agent.observations)))
            # agent.dump()
        # agent.drop_previous()
        agent.bad_kids_eaten_by_the_wolf() # kill bad kids

def check():
    episodic_data = do_episode_collect_trajectory(agent,env,max_steps=1000,render=True,feed=False,realtime=True)
    print('length of episode:',len(episodic_data[0]))
    print('total reward of episode:',np.sum(episodic_data[2]))
