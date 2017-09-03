# Proximal Policy Optimization algorithm
# implemented by Qin Yongliang

# this is intended to be run on py3.5
# rewritten with openai/baselines as a reference
import tensorflow as tf
import numpy as np
from canton import *
import gym

# To improve our policy via PPO, we must be able to parametrize and sample from it as a probabilistic distribution. A typical choice for continuous domain problems is the Diagonal Gaussian Distribution. It's simpler (and thus less powerful) than a full Multivariate Gaussian, but should work just fine.

# Knowledge of probabilistics is required to read and understand following code.

# DiagGaussian layer
# samples a diagonal gaussian distribution given input mean and logstd
# original version is in baselines/common/distributions.py
class DiagGaussian(can):
    def mode(self):
        return self.mean

    # -log(P(x)) given x
    def neglogp(self, x):
        return 0.5 * tf.sum(((x-self.mean)/self.std)**2, axis=-1) \
        + 0.5 * np.log(2*np.pi) * tf.to_float(tf.shape(x)[-1]) \
        + tf.sum(self.logstd, axis=-1)

    def logp(self, x):
        return - self.neglogp(x)

    def kl(self, p):
        # KL divergence between this distribution and another distribution
        assert isinstance(p, DiagGaussian)
        return tf.sum(p.logstd - self.logstd \
        + (self.std**2 + (self.mean-p.mean)**2)/(2.0 * p.std**2)-0.5, axis=-1)

    def entropy(self):
        raise NotImplementedError('bored')

    def sample(self): # sampling from
        return tf.random_normal(tf.shape(self.mean), mean=self.mean\
        stddev=self.std)

    def __call__(self, x):
        [self.mean, self.logstd] = x
        self.std = tf.exp(self.logstd)
        return self.sample(), self.mean
        # stochastically and deterministically generated actions respectively.

# this is a dual-output dense layer as it outputs the mean and logstd of a diagonal gaussian distribution.
# Put it between a hidden layer and a DiagGaussian layer.
class DiagGaussianParametrizer(can):
    def __init__(self,din,dout):
        self.Wmean = self.make_weight(din,dout,mean=0,stddev=1e-2)
        self.Bmean = self.make_bias(din,dout)
        self.Wlogstd = self.make_weight(din,dout,mean=0,stddev=1e-4)
        self.Blogstd = self.make_bias(din,dout)

    def __call__(self,x):
        mean = tf.matmul(x,self.Wmean) + self.Bmean
        logstd = tf.matmul(x,self.Wlogstd) + self.Blogstd
        return [mean,logstd]

# a simple MLP policy.
class Policy():
    def __init__(self, ob_space, ac_space):

        # 1. assume probability distribution is continuous
        assert len(ac_space.shape) == 1
        ac_dims = ac_space.shape[0]
        ob_dims = ob_space.shape[0]

        # 2. build our action network
        rect = Act('tanh')
        # apparently John doesn't give a fuck about ReLUs. Change above line as you wish.

        c = Can()
        c.add(Dense(ob_dims, 64))
        c.add(rect)
        c.add(Dense(64, 64))
        c.add(rect)
        c.add(DiagGaussianParametrizer(64, ac_dims))
        self.pd = c.add(DiagGaussian())
        c.chain()
        self.actor = c

        # 3. build our value network
        c = Can()
        c.add(Dense(ob_dims, 64))
        c.add(rect)
        c.add(Dense(64, 64))
        c.add(rect)
        c.add(Dense(64, 1))
        c.chain()
        self.critic = c

        # 4. build our action sampler
        input_state = ph([None])
        stochastic_action, deterministic_action = self.actor(input_state)
        value_prediction = self.critic(input_state)

        self.vpred = value_prediction

        def _act(state, stochastic=True):
            # assume state is of shape [1, dims]
            sess = get_session()
            res = sess.run([
                stochastic_action,
                deterministic_action,
                value_prediction
            ], feed_dict={input_state: state})

            sa, da, vp = res
            sa = sa[0]
            da = da[0]
            vp = vp[:,0]

            return sa,da,vp

        self.act = _act

# run a bunch of episodes and collect some trajectories.
def collect_trajectories(env, policy):
    print('collecting trajectory...')

    # things we have to collect
    collected = {
        's1':[], # observations before action
        'vp1':[], # value function prediction, given s1
        'a1':[], # action taken
        'r1':[], # reward received
        'done':[], # is the episode done after a1
        's2':[], # next observation
    }

    episodes = 30

    for e in range(episodes):
        print('collecting episode {}/{}'.format(e+1,episodes))

        episode_total_reward = 0
        episode_length = 0

        # initial observation
        ob = env.reset()
        while 1:
            # sample action from given policy
            sto_action, det_action, val_pred = policy.act(ob)

            # step to get reward
            new_ob, reward, done, info = env.step(sto_action)

            # append data into collection
            collected['s1'].append(ob)
            collected['vp1'].append(val_pred)
            collected['a1'].append(sto_action)
            collected['r1'].append(reward)
            collected['done'].append(1 if done else 0)
            collected['s2'].append(new_ob)

            ob = new_ob # assign new_ob to prev ob

            episode_total_reward+=reward
            episode_length+=1

            if isdone or episode_length>=200:
                # episode is done, either natually or forcifully
                collected['done'][-1] = 1
                print('episode {} done in {} steps, total reward:{}'.format(
                    e+1, episode_length, episode_total_reward,
                ))
                break

    return collected

# estimate target value (which we are trying to make our critic to fit) and advantage with GAE(lambda) from collected trajectories
def append_vtarg_and_adv(collected):
    # you know what these mean, don't you?
    gamma = 0.99
    lam = 0.95

    s1 = collected['s1']
    vp1 = collected['vp1']
    a1 = collected['a1']
    r1 = collected['r1']
    done = collected['done']
    s2 = collected['s2']

    T = len(s1)

    advantage = [None]*T

    last_adv = 0
    for t in reversed(range(T)): # step T-1, T-2 ... 0
        # whether a given step is terminating step
        nonterminal = 1 - done[t]

        # delta = (reward_now) + (predicted_future_t+1) - (predicted_future_t)
        delta = r1[t] + gamma * vp1[t+1] * nonterminal - vp1[t]

        advantage[t] = delta + gamma * lam * nonterminal * last_adv
        last_adv = advantage[t]

    collected['advantage'] = 
    collected['tdlamret'] =


def learn():
    # get swingy
    envname = 'Pendulum-v0'
    env = gym.make(envname)

    ob_space = env.observation_space
    ac_space = env.action_space

    # improve policy w.r.t. old_policy
    policy = Policy(ob_space, ac_space)
    old_policy = Policy(ob_space, ac_space)

    adv_target = ph([None]) # Target advantage function
    ret = ph([None]) # Empirical return

    # lrmult is not implemented.

    clip_param = 0.2 # magical epsilon in paper

    # observations and actions will be sampled from episodes.
    obs, actions = ph([None]), ph([None])

    # P_new(actions)/P_old(actions)
    ratio = tf.exp(policy.pd.logp(actions) - old_policy.pd.logp(actions))

    surr1 = ratio * adv_target
    surr2 = tf.clip(ratio, 1.0-clip_param, 1.0+clip_param) * adv_target

    policy_surrogate = - tf.mean(tf.minimum(surr1,surr2)) # L^CLIP in paper

    value_loss = tf.mean((policy.vpred-ret)**2) # how bad is our critic?

    total_loss = policy_surrogate + value_loss # original implementation used this total_loss

    opt = tf.train.optimizer.Adam(1e-4)

    actor_trainstep = opt.minimize(policy_surrogate, var_list=self.actor.get_weights())
    critic_trainstep = opt.minimize(value_loss, var_list=self.critic.get_weights())

    # after one policy iteration, assign old_policy to current policy.
    def assign_old_eq_new():
        ops = [tf.assign(o,n) for o,n in zip(old_policy.actor.get_weights(), policy.actor.get_weights())]
        ops += [tf.assign(o,n) for o,n in zip(old_policy.critic.get_weights(), policy.critic.get_weights())]
        get_session().run([ops])

    get_session().run(gvi()) # init global variables for TF

    def r(iters=2):
        print('start running')
        for i in iters:
            print('iteration {}'.format(i))

            # 0. assign new to old
            assign_old_eq_new()

            # 1. collect trajectories w/ current policy
            collected = collect_trajectories(env, policy)

            # 2. estimate value target and advantage from collected trajectories
            append_vtarg_and_adv(collected)
