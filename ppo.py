# Proximal Policy Optimization algorithm
# implemented by Qin Yongliang
# rewritten with openai/baselines as a reference

# this is intended to be run on py3.5

# terminology
# ob, obs, input, observation, state, s1 : observation in RL
# ac, action, a1, output : action in RL
# reward, return, ret, r1, tdlamret : (per step / expected future) reward in RL
# adv, advantage, adv_target, atarg : advantage (estimated / target) in RL

import tensorflow as tf
import numpy as np
from canton import *
import gym

# low-passed gaussian noise to help with exploration.
from gaussian import lowpassgaussian as lpgs

# To improve our policy via PPO, we must be able to parametrize and sample from it as a probabilistic distribution. A typical choice for continuous domain problems is the Diagonal Gaussian Distribution. It's simpler (and thus less powerful) than a full Multivariate Gaussian, but should work just fine.

# Knowledge of probabilistics is required to read and comprehend following code.

# DiagGaussian layer
# sample from a diagonal gaussian distribution given input mean and logstd
# original version is in baselines/common/distributions.py
class DiagGaussian(Can):
    # -log(P(x)) given P and x
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(((x-self.mean)/self.std)**2, axis=-1) \
        + 0.5 * np.log(2*np.pi) * tf.to_float(tf.shape(x)[-1]) \
        + tf.reduce_sum(self.logstd, axis=-1)

    def logp(self, x):
        return - self.neglogp(x)

    # def kl(self, p):
    #     # KL divergence between this distribution and another distribution
    #     assert isinstance(p, DiagGaussian)
    #     return tf.reduce_sum(p.logstd - self.logstd \
    #     + (self.std**2 + (self.mean-p.mean)**2)/(2.0 * p.std**2)-0.5, axis=-1)

    def entropy(self):
        raise NotImplementedError('bored')

    # def sample(self): # sampling from
    #     return tf.random_normal(tf.shape(self.mean), mean=self.mean, stddev=self.std)

    def __call__(self, x):
        [self.mean, self.logstd] = x
        self.std = tf.exp(self.logstd)
        return self.mean, self.std
        # stochastically and deterministically generated actions respectively.

# this is a dual-output dense layer as it outputs the mean and logstd of a diagonal gaussian distribution.
# Put it between a hidden layer and a DiagGaussian layer.
class DiagGaussianParametrizer(Can):
    def __init__(self,din,dout):
        super().__init__()
        # output amplitude is discounted to center the distributions on start.
        self.mean_layer = self.add(Dense(din,dout,stddev=1e-2))
        self.logstd_layer = self.add(Dense(din,dout,stddev=1e-2,mean=2.))
        # the output will have a logstd of 2, or std of e^2. Good for exploration

    def __call__(self,x):
        mean, logstd = self.mean_layer(x), self.logstd_layer(x)
        return [mean, logstd]

# a simple MLP policy.
class Policy(Can):
    def __init__(self, ob_space, ac_space):
        super().__init__()

        # 1. assume probability distribution is continuous
        assert len(ac_space.shape) == 1
        self.ac_dims = ac_dims = ac_space.shape[0]
        self.ob_dims = ob_dims = ob_space.shape[0]

        # 2. build our action network
        rect = Act('tanh')
        # apparently John doesn't give a fuck about ReLUs. Change above line as you wish.

        c = Can()
        c.add(Dense(ob_dims, 256, stddev=1))
        c.add(rect)
        c.add(Dense(256, 64, stddev=1))
        c.add(rect)
        c.add(DiagGaussianParametrizer(64, ac_dims))
        self.dg = c.add(DiagGaussian())
        c.chain()
        self.actor = self.add(c)

        # 3. build our value network
        c = Can()
        c.add(Dense(ob_dims, 256, stddev=1))
        c.add(rect)
        c.add(Dense(256, 64, stddev=1))
        c.add(rect)
        c.add(Dense(64, 1, stddev=1))
        c.chain()
        self.critic = self.add(c)

# our PPO agent.
class ppo_agent:
    def __init__(
        self, ob_space, ac_space,
        horizon=2048,
        gamma=0.99, lam=0.95,
        train_epochs=10, batch_size=64,
        ):
        self.current_policy = Policy(ob_space, ac_space)
        self.old_policy = Policy(ob_space, ac_space)
        self.current_policy.actor.summary()
        self.current_policy.critic.summary()

        self.gamma, self.lam, self.horizon = gamma, lam, horizon
        self.train_epochs, self.batch_size = train_epochs, batch_size

        self.act, self.train_for_one_step, self.assign_old_eq_new = self.build_functions()

        from plotter import interprocess_plotter as plotter
        self.plotter = plotter(2)

    # build graph and actions for training with tensorflow.
    def build_functions(self):
        # the 'lrmult' parameter is not implemented.

        # improve policy w.r.t. old_policy
        policy, old_policy = self.current_policy, self.old_policy

        # Input Placeholders
        states, actions = ph([None]), ph([None]) # you know these two
        adv_target = ph([None]) # Target advantage function, estimated
        ret = ph([None]) # Empirical return, estimated

        # run observation thru the networks
        policy_mean, policy_std = policy.actor(states)
        policy_val_pred = policy.critic(states)

        old_policy_mean, old_policy_std = old_policy.actor(states)
        old_policy_val_pred = old_policy.critic(states)

        # ratio = P_now(state, action) / P_old(state, action)
        # state is previously connected so we will pass in actions only
        ratio = tf.exp(policy.dg.logp(actions) - old_policy.dg.logp(actions))

        # surr1 -> policy gradient
        surr1 = ratio * adv_target

        # surr2 -> policy deviation
        clip_param = 0.2 # magical epsilon in paper
        surr2 = tf.clip_by_value(ratio, 1.0-clip_param, 1.0+clip_param) * adv_target

        # together they form the L^CLIP loss in PPO paper.
        policy_surrogate = - tf.reduce_mean(tf.minimum(surr1,surr2))

        # how far is our critic's prediction from estimated return?
        value_prediction = policy_val_pred
        value_loss = tf.reduce_mean((value_prediction-ret)**2)

        # learn the actor more slowly to maximize exploration
        opt_a = tf.train.AdamOptimizer(1e-4)
        opt_c = tf.train.AdamOptimizer(3e-4)

        # # sum of two losses used in original implementation
        # total_loss = policy_surrogate + value_loss
        # combined_trainstep = opt.minimize(total_loss, var_list=policy.actor.get_weights()+policy.critic.get_weights())

        # I decided to go with the following instead
        actor_trainstep = opt_a.minimize(policy_surrogate, var_list=policy.actor.get_weights())
        critic_trainstep = opt_c.minimize(value_loss, var_list=policy.critic.get_weights())

        # 1. build our action sampler: given observation, generate action
        def act(state, stochastic=True):
            # assume state is ndarray of shape [dims]
            state = state.view()
            state.shape = (1,) + state.shape

            res = get_session().run([
                policy_mean,
                policy_std,
                value_prediction
            ], feed_dict={states: state})

            pm, ps, vp = res
            # [batch, dims] [batch, dims] [batch, 1]
            return pm[0], ps[0], vp[0,0]

        # update current policy, given sampled trajectories.
        def train_for_one_step(_states, _actions, _adv_target, _ret):
            # [print(a.shape) for a in [_obs,_actions,_adv_target,_ret]]
            res = get_session().run(
                [ # perform training and collect losses in one go
                    policy_surrogate, value_loss,
                    actor_trainstep, critic_trainstep,
                    # combined_trainstep,
                ],
                feed_dict = {
                    states:_states, actions:_actions,
                    adv_target:_adv_target, ret:_ret,
                }
            )
            # res[0] is ploss, res[1] is val_loss
            return res

        # assign old_policy's weights equal to current policy.
        assign_ops = [tf.assign(o,n) for o,n in zip(old_policy.get_weights(), policy.get_weights())]
        print('total of {} weights to assign from new to old'.format(len(assign_ops)))
        def assign_old_eq_new():
            get_session().run([assign_ops])

        return act, train_for_one_step, assign_old_eq_new

    # run a bunch of episodes with current_policy on env and collect some trajectories.
    def collect_trajectories(self, env):
        policy = self.current_policy
        horizon = self.horizon
        print('collecting trajectory...')

        # things we have to collect
        collected = {
            's1':[], # observations before action
            'vp1':[], # value function prediction, given s1
            'a1':[], # action taken
            'r1':[], # reward received
            'done':[], # is the episode done after a1
            # 's2':[], # next observation
        }

        # minimum length we are going to collect
        # horizon = 2048

        # counters
        ep = 0
        steps = 0
        sum_reward = 0

        # initialize noise sources
        # this noise has a stddev of 1 but has 1/f^2 spectral density
        # better exploration than pure gaussian
        ns = [lpgs() for _ in range(policy.ac_dims)]
        def noise_sample():
            return np.array([n.sample() for n in ns], dtype='float32')

        while 1:
            # print('collecting episode {}'.format(ep+1), end='\r')

            episode_total_reward = 0
            episode_length = 0

            # initial observation
            ob = env.reset()
            while 1:
                # sample action from given policy
                mean, std, val_pred = self.act(ob)
                sto_action = noise_sample() * std + mean

                # step environment with action and obtain reward
                new_ob, reward, done, info = env.step(sto_action)

                # append data into collection
                collected['s1'].append(ob)
                collected['vp1'].append(val_pred)
                collected['a1'].append(sto_action)
                collected['r1'].append(reward) # downscaled reward
                collected['done'].append(1 if done else 0)
                # collected['s2'].append(new_ob)

                ob = new_ob # assign new_ob to prev ob

                # counting
                episode_total_reward+=reward
                episode_length+=1
                steps+=1

                # if episode is done, either natually or forcifully
                if done or episode_length >= 1000:
                    collected['done'][-1] = 1
                    print('episode {} done in {} steps, total reward:{}'.format(
                        ep+1, episode_length, episode_total_reward,
                    ))
                    self.plotter.pushys([ep,episode_total_reward])
                    break

            sum_reward += episode_total_reward
            print('{}/{} steps collected in {} episode(s)'.format(steps,horizon,ep+1), end='\r')
            if steps>= horizon:
                break
            else:
                ep+=1

        print('mean reward per episode:{}'.format(sum_reward/(ep+1)))
        return collected

    # estimate target value (which we are trying to make our critic to fit) via TD(lambda), and advantage using GAE(lambda), from collected trajectories.
    def append_vtarg_and_adv(self, collected):
        # you know what these mean, don't you?
        gamma = self.gamma # 0.99
        lam = self.lam # 0.95

        s1 = collected['s1']
        vp1 = collected['vp1']
        a1 = collected['a1']
        r1 = collected['r1']
        done = collected['done']
        # s2 = collected['s2']

        T = len(s1)
        advantage = [None]*T

        last_adv = 0
        for t in reversed(range(T)): # step T-1, T-2 ... 0
            # delta = (reward_now) + (predicted_future_t+1) - (predicted_future_t)
            delta = r1[t] + (0 if done[t] else gamma * vp1[t+1]) - vp1[t]

            advantage[t] = delta + gamma * lam * (1-done[t]) * last_adv
            last_adv = advantage[t]

        collected['advantage'] = advantage
        collected['tdlamret'] = [a+v for a,v in zip(advantage, vp1)]

    # perform one policy iteration
    def iterate_once(self, env):

        # 0. assign new to old
        self.assign_old_eq_new()

        # 1. collect trajectories w/ current policy
        collected = self.collect_trajectories(env)

        # 2. estimate value target and advantage from collected trajectories
        self.append_vtarg_and_adv(collected)

        # 3. data processing
        # shuffling
        indices = np.arange(len(collected['s1']))
        np.random.shuffle(indices)

        # numpyization
        ob, ac, atarg, tdlamret = [
            np.take(np.array(collected[k]).astype('float32'), indices, axis=0)
            for k in ['s1', 'a1', 'advantage', 'tdlamret']
        ]

        # expand dimension for minibatch training
        for nd in [ob,ac,atarg,tdlamret]:
            if nd.ndim == 1:
                nd.shape += (1,)

        # standarize/normalize
        atarg = (atarg - atarg.mean())/atarg.std()

        # 4. train for some epochs
        train_epochs = self.train_epochs # 30
        batch_size = self.batch_size # 512
        import time
        lasttimestamp = time.time()

        for e in range(train_epochs):
            for j in range(0, len(ob)-batch_size+1, batch_size): # ignore tail
                res = self.train_for_one_step(
                    ob[j:j+batch_size],
                    ac[j:j+batch_size],
                    atarg[j:j+batch_size],
                    tdlamret[j:j+batch_size],
                )
                ploss, vloss = res[0],res[1]
                if time.time() - lasttimestamp > 0.2:
                    lasttimestamp = time.time()
                    print(' '*30, 'ploss: {:6.4f} vloss: {:6.4f}'.format(
                        ploss, vloss),end='\r')
        print('iteration done.')

if __name__ == '__main__':
    # get swingy
    envname = 'Pendulum-v0'
    envname = 'BipedalWalker-v2'
    env = gym.make(envname)

    agent = ppo_agent(
        env.observation_space, env.action_space,
        horizon=8192,
        gamma=0.995,
        lam=0.98,
        train_epochs=20,
        batch_size=128,
    )

    get_session().run(gvi()) # init global variables for TF

    def r(iters=2):
        print('start running')
        for i in range(iters):
            print('optimization iteration {}/{}'.format(i+1, iters))
            agent.iterate_once(env)
    r(50)
