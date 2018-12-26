import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import gym
from gym import wrappers

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

class HiddenLayer:
    
    def __init__(self, Mi, Mo, f=tf.nn.relu):
        self.W = tf.Variable(tf.random_normal((Mi, Mo)))
        self.b = tf.Variable(np.zeros((1, Mo)).astype(np.float32))
        self.f = f
        self.params = [self.W, self.b]
        
    def forward(self, X):
        output = self.f(tf.matmul(X, self.W) + self.b)
        return output       
        
class DQN:
    
    def __init__(self, input_dims, hidden_layers, output_dims, max_exp=10000, min_exp=100, batch_sz=32):
        self.D = input_dims
        self.K = output_dims
        self.min_exp = min_exp
        self.max_exp = max_exp
        self.batch_sz = batch_sz
        self.layers = []
        self.params = []
        layer = HiddenLayer(self.D, hidden_layers[0])
        self.layers.append(layer)
        for i in range(len(hidden_layers)-1):
            layer = HiddenLayer(hidden_layers[i], hidden_layers[i+1])
            self.layers.append(layer)
        layer = HiddenLayer(hidden_layers[-1], self.K, tf.nn.softmax)
        self.layers.append(layer)
        for layer in self.layers:
            self.params += layer.params
        self.replay_buffer = {'s':[], 'a':[], 'r':[], 's_p':[], 'done':[]}
        self.X = tf.placeholder(tf.float32, (None, self.D))
        self.G = tf.placeholder(tf.float32, (None,))
        self.A = tf.placeholder(tf.int32, (None,))
        out = self.X
        for layer in self.layers:
            out = layer.forward(out)
        yhat = out
        self.predict_op = yhat
        selected_action_values = tf.reduce_sum(self.predict_op*tf.one_hot(self.A, self.K), 
                                               reduction_indices=[1])
        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
        
    def train(self, target_network, gamma):
        if len(self.replay_buffer['s']) < self.min_exp:
            return
        idx = np.random.choice(len(self.replay_buffer['s']), self.batch_sz, False)
        states = [self.replay_buffer['s'][i] for i in idx]
        actions = [self.replay_buffer['a'][i] for i in idx]
        next_states = [self.replay_buffer['s_p'][i] for i in idx]
        rewards = [self.replay_buffer['r'][i] for i in idx]
        dones = [self.replay_buffer['done'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = []
        for next_q, r, done in zip(next_Q, rewards, dones):
            if done:
                targets.append(r)
            else:
                Qval = r + gamma*next_q
                targets.append(Qval)
        self.session.run(self.train_op, {self.X:states, self.G:targets, self.A:actions})
        
    def predict(self, states):
        states = np.atleast_2d(states)
        output = self.session.run(self.predict_op, {self.X:states})
        return output
    
    def sample_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict(state)[0])
    
    def set_session(self, session):
        self.session = session
    
    def add_experience(self, s, a, r, s_p, done):
        if len(self.replay_buffer['s']) >= self.max_exp:
            self.replay_buffer['s'].pop(0)
            self.replay_buffer['a'].pop(0)
            self.replay_buffer['r'].pop(0)
            self.replay_buffer['s_p'].pop(0)
            self.replay_buffer['done'].pop(0)
        self.replay_buffer['s'].append(s)
        self.replay_buffer['a'].append(a)
        self.replay_buffer['s_p'].append(s_p)
        self.replay_buffer['r'].append(r)
        self.replay_buffer['done'].append(done)
    
    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)
        
def play_one(env, enet, tnet, gamma, eps, copy_period):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        action = enet.sample_action(observation, eps)
        prev_observation = observation
        #env.render()
        observation, reward, done, info = env.step(action)
        totalreward += reward
        enet.add_experience(prev_observation, action, reward, observation, done)
        enet.train(tnet, gamma)
        iters += 1
        if iters % copy_period == 0:
            tnet.copy_from(enet)
    #env.close()
    return totalreward

env = gym.make('MountainCar-v0')
env._max_episode_steps = 4000
gamma = 0.99
copy_period = 100
D = env.reset().shape[0]
K = env.action_space.n
enet = DQN(D, [200, 200], K)
tnet = DQN(D, [200, 200], K)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
enet.set_session(sess)
tnet.set_session(sess)
saver = tf.train.Saver()
saver.restore(sess, './mountaincardqn.ckpt')
if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
N = 10000
totalrewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(env, enet, tnet, gamma, eps, copy_period)
    totalrewards[n] = totalreward
    print("Total Reward on episode {}: {}".format(n, totalreward))
    if n % 100 == 0:
    	print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
    saver.save(sess, './mountaincardqn.ckpt')
print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()

plot_running_avg(totalrewards)