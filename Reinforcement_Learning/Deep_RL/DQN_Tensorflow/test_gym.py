import gym
from model import NN
import tensorflow as tf
import numpy as np
import random
import math


env = gym.make('Breakout-v0')

nn = NN()
nn2 = NN()
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

discount = 0.99
full_reward = []
total_reward = 0


def choose_action(Q):
    action = np.argmax(Q)
    if epsilon > random.random():
        action = env.action_space.sample()
    return action

print(env.action_space)

for i_episode in range(1, 2000):
    observation = env.reset()
    full_reward.append(total_reward)
    total_reward = 0
    actions_and_scores = []
    print discount
    epsilon = 0.01 + (0.99 / (i_episode))
    print epsilon
    #if discount < 0.89:
        #discount += 0.005
    loss = 0

    for t in range(1, 50000):
        Q = sess.run(nn.predict, feed_dict={nn.input: observation})
        action = choose_action(Q)
        env.render()
        # Take action, observe R y S'
        #for _ in range(4):
        observation1, reward, done, info = env.step(action)

        Q1 = sess.run(nn.predict, feed_dict={nn.input: observation1})



        total_reward += reward

        if done:
            Q[action] = total_reward
            loss += sess.run([nn.loss, nn.train], feed_dict={nn.input: observation, nn.targetQ: Q})[0]
            actions_and_scores.append({'Q': Q, 'obs': observation})
            break
        else:
            Q[action] = discount * (reward + np.max(Q1))
            loss += sess.run([nn.loss, nn.train], feed_dict={nn.input: observation, nn.targetQ: Q})[0]
            actions_and_scores.append({'Q': Q, 'obs': observation})
        observation = observation1

    for _ in range(20):
        t = actions_and_scores[int(math.floor(random.random() * len(actions_and_scores)))]
        sess.run([nn.loss, nn.train], feed_dict={nn.input: t['obs'], nn.targetQ: t['Q']})[0]


    print("total reward for episode " + str(i_episode) + " is " + str(total_reward))
    print "total loss for episode: {}".format(loss)
    #print("Creativity level: {}%".format(epsilon * 100))
    #print("Mastering level: {}%".format(discount * 100))
print("Full reward for this setup {}".format(full_reward))
t = np.arange(0.0, 400, 0.5)
