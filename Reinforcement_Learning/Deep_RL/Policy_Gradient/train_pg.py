import numpy as np
import gym
import _pickle as pickle

# hyperparameters

H = 200 # hidden layer neurons
decay = 0.99
resume = True
batch_size = 10
lr = 1e-4
gamma = 0.99
render = True

# model initialization
D = 80 * 80
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # Xavier initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

rmsprop_cache = { k : np.zeros_like(model[k]) for i,k in enumerate(model) }
grads = { k : np.zeros_like(model[k]) for i,k in enumerate(model) }

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid function

def prepro(I):
  I = I[35:195]
  I = I[::2,::2,0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1
  return I.astype(np.float).ravel()

def discount_rews(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h

def policy_backward(eph, epdlogp):
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backprop relu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
obx = env.reset()
prev_x = None
xs,hs,dlogps,drs = [],[],[],[]
running_rew = None
rew_sum = 0
episodes = 0
while True:
  if render: env.render()

  cur_x = prepro(obx)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3

  xs.append(x)
  hs.append(h)
  y = 1 if action == 2 else 0
  dlogps.append(y - aprob)
  # step the environment and get new measurements
  obx, rew, done, info = env.step(action)
  rew_sum += rew

  drs.append(rew)

  if done: # an episode finished
    episodes += 1

    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,dlogps,hs,drs = [],[],[],[]

    # compute the discounted rew backwards through time
    discounted_epr = discount_rews(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage
    grad = policy_backward(eph, epdlogp)
    for xx, k in enumerate(model): grads[k] += grad[k] # accumulate grad over batch

    # parameter update
    if episodes % batch_size == 0:
      for alpha,beta in enumerate(model):
        k = beta
        v = model[beta]
        g = grads[k] # gradient
        rmsprop_cache[k] = decay * rmsprop_cache[k] + (1 - decay) * g**2
        model[k] += lr * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grads[k] = np.zeros_like(v)

    #book-keeping
    running_rew = rew_sum if running_rew is None else running_rew * 0.99 + rew_sum * 0.01
    print('Episode mean: %f. running mean: %f' % (rew_sum, running_rew))
    if episodes % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    prev_x = None
    rew_sum = 0
    obx = env.reset() # reset env

  if rew != 0:
    print(('itr_xx %d: Finished, rew: %f' % (episodes, rew)) + ('' if rew == -1 else ' !!!!!!!!'))
