
import numpy as np
import matplotlib.pyplot as plt
import Environment
import actor_critic
import importlib
importlib.reload(Environment)

class Environment:
    def __init__(self, A_effect_vec, observable_features=(True, False, False), no_backtrack=True, MapString='', random_pit_prob=0.0, nR=0, nC=0, rStart=0, cStart=0, rTerm=0, cTerm=0, pit_vec=np.array([]), wall_vec=np.array([])):
        self.A_effect_vec = A_effect_vec
        self.nA = len(self.A_effect_vec)
        if not len(MapString) == 0:
            # Define world using a MapString
            MapLines = [r for r in MapString.split('\n') if len(r) > 0]
            tileCodes = np.array([np.fromstring(r, dtype=int, sep=' ') for r in MapLines])
            self.nR = tileCodes.shape[0]
            self.nC = tileCodes.shape[1]
            self.rStart = np.where(tileCodes == 1)[0][0]
            self.cStart = np.where(tileCodes == 1)[1][0]
            self.rTerm0 = np.where(tileCodes == 2)[0][0]
            self.cTerm0 = np.where(tileCodes == 2)[1][0]
            self.pit_vec, self.wall_vec = self.tileCodes_to_vectors(tileCodes)
        else:
            # Define world directly using variables
            self.nR = nR
            self.nC = nC
            self.rStart = rStart
            self.cStart = cStart
            self.rTerm0 = rTerm
            self.cTerm0 = cTerm
            self.pit_vec = pit_vec
            self.wall_vec = wall_vec
        self.s_r = 0
        self.s_c = 0
        self.rTerm = self.rTerm0
        self.cTerm = self.cTerm0
        self.pit_map = np.zeros((self.nR, self.nC))
        self.random_pit_prob = random_pit_prob
        self.create_wall_map()
        self.no_backtrack = no_backtrack
        self.mem_length = self.nR * self.nC
        self.memory = []
        self.observable_features = observable_features  # Coordinates; local pits; goal direction
        feature_vec, allowed_actions = self.state_to_features()
        self.nFeatures = int(np.prod(feature_vec.shape))
        self.run_checks()
    def run_checks(self):
        if self.rTerm0 == self.rStart and self.cTerm0 == self.cStart and not (np.isnan(self.rStart) or np.isnan(self.cStart)):
            raise Exception("Invalid environment definition: start and terminal points are the same.")
    def tileCodes_to_vectors(self, tileCodes):
        pit_vec = np.array([z for z in zip(np.where(tileCodes == 3)[0], np.where(tileCodes == 3)[1])])
        wall_vec = np.array([z for z in zip(np.where(tileCodes == 4)[0], np.where(tileCodes == 4)[1])])
        return pit_vec, wall_vec
    def create_wall_map(self):
        self.wall_map = np.zeros((self.nR, self.nC))
        for i_wall in range(self.wall_vec.shape[0]):
            self.wall_map[self.wall_vec[i_wall][0]][self.wall_vec[i_wall][1]] = 1
    def set_rewards(self, pit_punishment=-1, backtrack_punishment=-1, off_grid_punishment=-1, terminal_reward=0):
        self.pit_punishment = pit_punishment
        self.backtrack_punishment = backtrack_punishment
        self.off_grid_punishment = off_grid_punishment
        self.terminal_reward = terminal_reward
    def init_episode(self):
        self.s_r = self.rStart
        self.s_c = self.cStart
        while True:
            if np.isnan(self.rTerm0):
                self.rTerm = np.random.randint(0, self.nR)
            if np.isnan(self.cTerm0):
                self.cTerm = np.random.randint(0, self.nC)
            if np.isnan(self.rStart):
                self.s_r = np.random.randint(0, self.nR)
            if np.isnan(self.cStart):
                self.s_c = np.random.randint(0, self.nC)
            if self.s_r != self.rTerm or self.s_c != self.cTerm:
                break
        # Make pits: pre-set and random
        self.pit_map = np.zeros((self.nR, self.nC))
        for i_pit in range(self.pit_vec.shape[0]):
            r = self.pit_vec[i_pit][0]
            c = self.pit_vec[i_pit][1]
            if not (np.abs(r - self.rTerm) <= 1 and np.abs(c - self.cTerm) <= 1):
                self.pit_map[r][c] = 1
        for r in range(self.nR):
            for c in range(self.nC):
                die = np.random.rand()
                if die < self.random_pit_prob:
                    if not (np.abs(r - self.rTerm) <= 1 and np.abs(c - self.cTerm) <= 1):
                        self.pit_map[r, c] = 1
        self.memory = [(self.s_r, self.s_c) for n in range(self.mem_length)]
    def f_into_wall(self, new_s_r, new_s_c):
        in_wall = self.wall_map[new_s_r, new_s_c] == 1
        return in_wall
    def f_pit(self):
        in_pit = self.pit_map[self.s_r, self.s_c] == 1
        return in_pit
    def f_terminal(self):
        if self.s_r == self.rTerm and self.s_c == self.cTerm:
            return True
        else:
            return False
    def f_backtracking(self):
        if (self.s_r, self.s_c) in self.memory:
            return True
        else:
            return False
    def state_to_gridcoords(self):
        X = np.zeros((self.nR, self.nC))
        X[self.s_r, self.s_c] = 1
        X = X.reshape(self.nR * self.nC)
        return X
    def state_to_local(self):
        X = np.zeros((3, 3))
        for idr in range(3):
            dr = -1 + idr
            for idc in range(3):
                dc = -1 + idc
                r = self.s_r + dr
                c = self.s_c + dc
                if r >= 0 and r < self.nR and c >= 0 and c < self.nC:
                    if self.pit_map[r, c] == 1:
                        X[idr, idc] = 1
        X = X.reshape(np.prod(X.shape)).copy()
        return X
    def state_to_goalrelative(self):
        X = []
        dr = 0
        if self.s_r - self.rTerm > 0:
            dr = 1
        elif self.s_r - self.rTerm < 0:
            dr = -1
        dc = 0
        if self.s_c - self.cTerm > 0:
            dc = 1
        elif self.s_c - self.cTerm < 0:
            dc = -1
        for r in range(-1, 2):
            for c in range(-1, 2):
                 if r == dr and c == dc:
                    X.append(1)
                 else:
                    X.append(0)
        X = np.array(X)
        X = X.reshape(np.prod(X.shape)).copy()
        return X
    def get_observables_indices(self):
        a = 0
        b = self.nR * self.nC
        c = b
        d = b + 9
        e = d
        f = e + 9
        obs_ind = [(a, b), (c, d), (e, f)]
        return obs_ind
    def state_to_features(self):
        X_gridCoordinates = self.state_to_gridcoords()
        if self.observable_features[0] == False:
            X_gridCoordinates = 0 * X_gridCoordinates
        X_local = self.state_to_local()
        if self.observable_features[1] == False:
            X_local = 0 * X_local
        X_goal = self.state_to_goalrelative()
        if self.observable_features[2] == False:
            X_goal = 0 * X_goal
        # Create full feature vector (column)
        X = np.append(X_gridCoordinates, X_local)
        X = np.append(X, X_goal)
        X = X.reshape((len(X), 1))
        allowed_actions = np.array([])
        for a in range(len(self.A_effect_vec)):
            new_s_r = self.s_r + self.A_effect_vec[a][0]
            new_s_c = self.s_c + self.A_effect_vec[a][1]
            if (new_s_r >= 0 and new_s_r < self.nR) and (new_s_c >= 0 and new_s_c < self.nC) and not self.f_into_wall(new_s_r, new_s_c):
                if (self.no_backtrack == False) or (not (new_s_r, new_s_c) in self.memory):
                    allowed_actions = np.append(allowed_actions, a)
        return X, allowed_actions
    def respond_to_action(self, a):
        off_grid = False
        new_s_r = self.s_r + self.A_effect_vec[a][0]
        new_s_c = self.s_c + self.A_effect_vec[a][1]
        if new_s_c < 0 or new_s_c >= self.nC:
            off_grid = True
        new_s_c = np.min([self.nC - 1, np.max([0, new_s_c])])
        if new_s_r < 0 or new_s_r >= self.nR:
            off_grid = True
        self.s_r = np.min([self.nR - 1, np.max([0, new_s_r])])
        self.s_c = np.min([self.nC - 1, np.max([0, new_s_c])])
        r = -1
        terminal = False
        if self.f_terminal():
            r = self.terminal_reward
            terminal = True
        else:
            if off_grid:
                r = r + self.off_grid_punishment
            if self.f_pit():
                r = r + self.pit_punishment
            if self.f_backtracking():
                r = r + self.backtrack_punishment
        if self.mem_length > 0:
            self.memory.pop(0)
            self.memory.append((self.s_r, self.s_c))
        return r, terminal


class Critic:
    def __init__(self, nFeatures):
        self.w = np.zeros((nFeatures, 1))
        self.z = np.zeros((nFeatures, 1))
        self.alpha0 = 0.1
        self.lambda0 = 0.5
        self.gamma0 = 0.9
    def get_v(self, feature_vec):
        return np.sum(feature_vec * self.w)
    def delta_v(self, feature_vec):
        return feature_vec
    def update(self, r, feature_vec, feature_vec_new, terminal):
        if not terminal:
            self.delta0 = r + self.gamma0 * self.get_v(feature_vec_new) - self.get_v(feature_vec)
        else:
            self.delta0 = r - self.get_v(feature_vec)
        self.z = self.lambda0 * self.z + self.delta_v(feature_vec)
        self.w = self.w + self.alpha0 * self.delta0 * self.z
    def get_delta(self):
        return self.delta0

class Actor:
    def __init__(self, nFeatures, nA, action_error_prob=0.1):
        self.nFeatures = nFeatures
        self.nA = nA
        self.action_error_prob = action_error_prob
        self.theta0 = np.zeros((self.nFeatures, self.nA))
        self.z = np.zeros((self.nFeatures, self.nA))
        self.alpha0 = 0.5
        self.lambda0 = 0.5
        self.gamma0 = 0.9
        self.I = 1
        self.a = 0
    def policy_prob(self, feature_vec, b):
        c = np.max(np.dot(np.transpose(feature_vec), self.theta0))
        prefs = np.exp(np.dot(np.transpose(feature_vec), self.theta0) - c)
        prob = prefs[0][b] / np.sum(prefs)
        return prob
    def act_on_policy_q(self, feature_vec, allowed_actions=[], error_free=True):
        if len(allowed_actions) == 0:
            allowed_actions = np.array(range(self.nA))
        allowed_actions = allowed_actions.astype(int)
        action_error_rnd = np.random.rand()
        if action_error_rnd < self.action_error_prob and error_free == False:
            self.a = np.random.choice(allowed_actions)
        else:
            sa_q = np.array([])
            for b in allowed_actions:
                q = np.dot(feature_vec, self.theta0[:, b])
                sa_q = np.append(sa_q, q)
            self.a = allowed_actions[np.argmax(sa_q)]
        return self.a
    def act_on_policy_softmax(self, feature_vec, allowed_actions=[], error_free=True):
        if len(allowed_actions) == 0:
            allowed_actions = np.array(range(self.nA))
        allowed_actions = allowed_actions.astype(int)
        action_error_rnd = np.random.rand()
        if action_error_rnd < self.action_error_prob and error_free == False:
            print('random action')
            self.a = np.random.choice(allowed_actions)
        else:
            probs = np.array([])
            for b in range(self.nA):
                prob = self.policy_prob(feature_vec, b)
                probs = np.append(probs, prob)
            probs = probs[allowed_actions]
            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                #print('X-X-X\nX-X-X\nIllegal probs: ', probs, ', theta0: ', self.theta0, 'z: ', self.z, '\nX-X-X\nX-X-X\n')
                probs = np.ones(probs.shape) / len(probs) # If need to choose between effectively 0-prob allowed actions
            else:
                probs = probs / np.sum(probs)
            self.a = np.random.choice(allowed_actions, p=probs)
        return self.a
    def act_on_policy(self, feature_vec, allowed_actions=[], error_free=True):
        self.a = self.act_on_policy_softmax(feature_vec, allowed_actions, error_free)
        return self.a
    def delta_ln_pi(self, feature_vec):
        term1 = np.zeros((self.nFeatures, self.nA))
        iStates = np.where(feature_vec == 1)[0]
        term1[iStates, self.a] = 1
        term2 = np.zeros((self.nFeatures, self.nA))
        for b in range(self.nA):
            tmp = np.zeros((self.nFeatures, self.nA))
            tmp[iStates, b] = 1
            term2 = term2 + self.policy_prob(feature_vec, b) * tmp
        return term1 - term2
    def update(self, delta0, feature_vec):
        delta_this = self.delta_ln_pi(feature_vec)
        self.z = self.gamma0 * self.lambda0 * self.z + self.I * delta_this
        self.theta0 = self.theta0 + self.alpha0 * delta0 * self.z
        self.I = self.I * self.gamma0

class Agent:
    def __init__(self, nFeatures, nA):
        self.critic = Critic(nFeatures)
        self.actor = Actor(nFeatures, nA)
    def init_episode(self):
        self.critic.z = 0 * self.critic.z
        self.actor.z = 0 * self.actor.z
        self.actor.I = 1

class Simulation:
    def __init__(self, max_episode_length):
        self.ep_len = np.array([])
        self.max_episode_length = max_episode_length
        pass
    def train(self, nEpisodes, environment, agent):
        environment.init_episode()
        agent.init_episode()
        self.ep_len = np.array([])
        iEpisode = 0
        t_ep = 0
        while iEpisode < nEpisodes:
            print(iEpisode, '. ', end='', sep='')
            print('(', environment.s_r, ', ', environment.s_c, '). ', end='', sep='')
            feature_vec, allowed_actions = environment.state_to_features()
            print(allowed_actions, '. ', sep='', end='')
            a = agent.actor.act_on_policy(feature_vec, allowed_actions=allowed_actions, error_free=False)
            r, terminal = environment.respond_to_action(a)
            feature_vec_new, allowed_actions_new = environment.state_to_features()
            agent.critic.update(r, feature_vec, feature_vec_new, terminal)
            delta0 = agent.critic.get_delta()
            agent.actor.update(delta0, feature_vec)
            print('a = ', a, '. r = ', r, '. delta0 = ', delta0, ', max abs w = ', np.max(np.abs(agent.critic.w)), ', max abs theta0 = ', np.max(np.abs(agent.actor.theta0)), end='\n', sep='')
            if np.isnan(delta0):
                break
            if t_ep > self.max_episode_length:
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                print('Episode failed.')
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                terminal = True
            if terminal == True:
                environment.init_episode()
                agent.init_episode()
                self.ep_len = np.append(self.ep_len, t_ep)
                t_ep = 0
                iEpisode = iEpisode + 1
            t_ep = t_ep + 1
        return agent
    def test(self, environment, agent):
        environment.init_episode()
        terminal = False
        route = np.array([environment.s_r, environment.s_c])
        t = 0
        while not terminal:
            print(t, ': ', end='')
            feature_vec, allowed_actions = environment.state_to_features()
            a = agent.actor.act_on_policy(feature_vec, allowed_actions=allowed_actions, error_free=True)
            print('(', environment.s_r, environment.s_c, ')')
            r, terminal = environment.respond_to_action(a)
            if not terminal:
                route = np.append(route, [environment.s_r, environment.s_c])
            t = t + 1
        route = route.reshape(int(len(route)/2), 2)
        return route
    def plots(self, environment, agent, route):
        obs_ind = environment.get_observables_indices()
        W = agent.critic.w[obs_ind[0][0]:obs_ind[0][1]].reshape((environment.nR, environment.nC))
        W_local = np.max(agent.critic.w[obs_ind[1][0]:obs_ind[1][1]], axis=1).reshape(3, 3)
        W_goal = np.max(agent.critic.w[obs_ind[2][0]:obs_ind[2][1]], axis=1).reshape(3, 3)
        T_local = agent.actor.theta0[obs_ind[1][0]:obs_ind[1][1], :]
        T_goal = agent.actor.theta0[obs_ind[2][0]:obs_ind[2][1], :]
        figs, ax = plt.subplots(4, 3)
        ax[0, 0].plot(self.ep_len)
        ax[0, 1].pcolormesh(W)
        ax[1, 0].pcolormesh(W_local)
        ax[1, 1].pcolormesh(W_goal)
        ax[0, 2].pcolormesh(environment.pit_map + environment.wall_map * 2)
        if len(route) > 0:
            ax[0, 2].scatter(route[:, 1] + 0.5, route[:, 0] + 0.5)
            ax[0, 2].plot(route[:, 1] + 0.5, route[:, 0] + 0.5)
            ax[0, 2].xaxis.set_ticks(ticks=np.array([range(environment.nC)]).reshape(environment.nC) + 0.5, labels=np.array([range(environment.nC)]).reshape(environment.nC))
            ax[0, 2].yaxis.set_ticks(ticks=np.array([range(environment.nR)]).reshape(environment.nR) + 0.5, labels=np.array([range(environment.nR)]).reshape(environment.nR))
        ax[2,0].pcolormesh(T_local)
        ax[2,2].pcolormesh(T_goal)
        plt.show()

#
# Basic search in 2D GridWorld; the observable feature of the Environment is the state itself, i.e., the location coordinates.
#
# A_effect_vec defines the movement caused by actions.
# MapString defines the world: 0 = default state, 1 = start, 2 = terminal, 3 = pit, 4 = wall.

A_effect_vec = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
MapString = '''
4 0 0 0 0 0 0 0 0 0
0 0 0 0 0 4 2 0 0 0
0 0 0 0 0 4 4 4 4 0
0 0 0 0 0 0 0 0 0 0
3 3 3 3 0 3 3 0 0 0
0 1 0 0 0 0 0 0 0 0
'''
random_pit_prob = 0.0
pit_punishment = -2
backtrack_punishment = 0
off_grid_punishment = -1
terminal_reward = 0
observable_features = (True, False, False)

environment = Environment(A_effect_vec, observable_features, MapString=MapString, random_pit_prob=random_pit_prob)
environment.set_rewards(pit_punishment=pit_punishment, backtrack_punishment=backtrack_punishment, off_grid_punishment=off_grid_punishment, terminal_reward=terminal_reward)

agent = teg_actorCritic.Agent(environment.nFeatures, environment.nA)
agent.critic.lamba0 = 0.5
agent.actor.lamba0 = 0.5

max_episode_length = 1e6
sim = teg_actorCritic.Simulation(max_episode_length)

agent = sim.train(1e3, environment, agent)
route = sim.test(environment, agent)
sim.plots(environment, agent, route)

#
# Random environment per episode (i.e., starting, terminal, and pit locations), relative environment features to learn
#
nR = 6; nC = 10
rStart = np.nan; cStart = np.nan;
rTerm = np.nan; cTerm = np.nan
A_effect_vec = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
random_pit_prob = 0.2
pit_punishment = -1
backtrack_punishment = 0
off_grid_punishment = -1
terminal_reward = 0
observable_features = (False, True, True)

environment = Environment.Environment(A_effect_vec, observable_features, nR=nR, nC=nR, rStart=rStart, cStart=cStart, rTerm=rTerm, cTerm=cTerm, random_pit_prob=random_pit_prob)
environment.set_rewards(pit_punishment=pit_punishment, backtrack_punishment=backtrack_punishment, off_grid_punishment=off_grid_punishment, terminal_reward=terminal_reward)

agent = teg_actorCritic.Agent(environment.nFeatures, environment.nA)
agent.critic.lamba0 = 0
agent.actor.lamba0 = 0

max_episode_length = 1e6
sim = teg_actorCritic.Simulation(max_episode_length)

agent = sim.train(1e3, environment, agent)
route = sim.test(environment, agent)
sim.plots(environment, agent, route)