# code from http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html

import numpy as np
import gym
from gym.spaces import Discrete, Box

def elu(x):
    return (x>0) * x + (x<0) * (1 - np.exp(x))

# ================================================================
# Policies
# ================================================================

class DeterministicDiscreteActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n


        imd = 4
        r = dim_ob * imd
        rn = r + imd

        r2 = imd * n_actions + rn
        rn2 = r2 + n_actions

        assert len(theta) == (dim_ob+1)*imd + (imd+1)*n_actions

        self.W = theta[0 : r].reshape(dim_ob, imd)
        self.b = theta[r : rn ].reshape(1, imd)

        self.W2 = theta[rn : r2].reshape(imd, n_actions)
        self.b2 = theta[r2 : rn2].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        if not hasattr(self,'ob'):
            self.ob = ob
        if not hasattr(self,'itg_ob'):
            self.itg_ob = 0

        delta_ob = ob - self.ob
        self.itg_ob = self.itg_ob * .8 + delta_ob * .2

        y = ob.dot(self.W) + self.b
        imd = elu(y)

        yd = delta_ob.dot(self.W) + self.b
        imd -= elu(yd)

        # yitg = self.itg_ob.dot(self.W) + self.b
        # imd += elu(yitg)

        # expd = np.exp(delta_ob).dot(self.W) + self.b
        # imd += np.tanh(expd)

        y2 = imd.dot(self.W2)+self.b2
        a = y2.argmax()

        self.ob = ob
        return a

class DeterministicContinuousActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0 : dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac : None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

def do_episode(policy, env, num_steps, render=True):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%10==0: env.render()
        if done : break
    return total_rew

env = None
def noisy_evaluation(theta):
    policy = make_policy(theta)
    rew = do_episode(policy, env, num_steps)
    print('gotrew',rew)
    return rew

def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DeterministicDiscreteActionLinearPolicy(theta,
            env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return DeterministicContinuousActionLinearPolicy(theta,
            env.observation_space, env.action_space)
    else:
        raise NotImplementedError

# Task settings:
env = gym.make('Acrobot-v1') # Change as needed
num_steps = 1000 # maximum length of episode
# Alg settings:
n_iter = 100 # number of iterations of CEM
batch_size = 50# number of samples per batch
elite_frac = 0.1 # fraction of samples used as elite set

if isinstance(env.action_space, Discrete):
    imd = 4
    dim_theta = (env.observation_space.shape[0]+1) * imd + (imd+1) * env.action_space.n
elif isinstance(env.action_space, Box):
    dim_theta = (env.observation_space.shape[0]+1) * env.action_space.shape[0]
else:
    raise NotImplementedError

# Initialize mean and standard deviation
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)*1

print(theta_mean.shape,theta_std.shape)

def r(n_iter=10,render=True):
    # Now, for the algorithm
    for iteration in range(n_iter):
        # Sample parameter vectors
        thetas = []

        #sample thetas from mean and std
        for i in range(batch_size):
            theta = theta_mean.copy()
            for k in range(len(theta)):
                theta[k] = np.random.normal(loc=theta_mean[k],scale=theta_std[k])
            thetas.append(theta)

        rewards = [noisy_evaluation(theta) for theta in thetas]
        # Get elite parameters
        n_elite = int(batch_size * elite_frac)
        elite_inds = np.argsort(rewards)[batch_size - n_elite:batch_size]
        elite_thetas = [thetas[i] for i in elite_inds]
        # Update theta_mean, theta_std

        global theta_mean,theta_std
        theta_mean = np.mean(elite_thetas,axis=0)
        theta_std = np.mean(np.power(elite_thetas - theta_mean,2), axis=0) ** .5 

        print(theta_mean.shape,theta_std.shape,np.min(theta_std))

        # theta_mean = YOUR_CODE_HERE
        # theta_std = YOUR_CODE_HERE

        print(
        "iteration {}. mean rew f: {:.3f}. max rew f: {:.3f}".format(
        iteration, np.mean(rewards), np.max(rewards)
        )
        )
        do_episode(make_policy(theta_mean), env, num_steps, render=render)
