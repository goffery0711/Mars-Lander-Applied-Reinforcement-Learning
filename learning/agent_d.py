'''
File contains different classes that handle reinforcement learning implementing Sarsa and Sarsa lambda
'''
# imports
import numpy as np
import learning.plotting
import logging

from collections import defaultdict
from abc import ABCMeta, abstractmethod


SCALE = 5


class State_estimator():
    '''
    Class to encode the continous features to discrete values, we achive this by putting a grid on the enviroment.
    '''
    def __init__(self):
        # init q table and eglibility trace
        self.q_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        # self.eg_trace = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.eg_trace = []

    def __feature_construction(self, state):
        '''
        Method that transforms the features from x and y values to more interesting values like the distance to goal
        @param state : state observation returned by the enviroment
        @return : selected states to define the grid.
        '''
        # take the position of the lander
        return_states = []
        return_states.append(state[0])
        return_states.append(state[1])

        return return_states

    def __get_discrete_state(self, state):
        '''
        Discretization of the state values.
        @param state : reduced state space
        @return : disctreetized states
        '''
        state = state * SCALE
        state = self.__feature_construction(state)
        d_state = []

        for val in state:
            d_state.append(int(val))

        return d_state

    def get_eglibility_trace(self, state, action):
        '''
        Returns the value of the eligibility trace for state, action pair
        '''
        state = self.__get_discrete_state(state)
        return self.eg_trace[state[0]][state[1]][action]

    def update_eglibility_trace(self, state, action, value):
        '''
        Sets the value of the eligibility trace at position state, action to value
        '''
        state = self.__get_discrete_state(state)
        self.eg_trace[state[0]][state[1]][action] = value

    def get_q_value(self, state, action=None):
        '''
        Returns the Q value for state, action pair
        '''
        state = self.__get_discrete_state(state)

        if action is None:
            q_ret = []
            q_0 = self.q_table[state[0]][state[1]][0]
            q_ret.append(q_0)
            q_ret.append(self.q_table[state[0]][state[1]][1])

        else:
            assert action == 0 or action == 1
            q_ret = self.q_table[state[0]][state[1]][action]

        return q_ret

    def update_q_value(self, state, action, value):
        '''
        Updates Q tabel entry at position state, action to value
        '''
        state = self.__get_discrete_state(state)

        self.q_table[state[0]][state[1]][action] = value


class Simple_agent(metaclass=ABCMeta):
    '''
    Abstract class defining basic methods and attributes for learning.
    '''
    def __init__(self, env, logfile_name, logging_level):
        logging.basicConfig(filename=logfile_name, level=logging_level)
        self.env = env
        self.estimator = State_estimator()

    def get_epsilon_greedy_policy(self, epsilon):
        '''
        Method returning an epsilon greedy policy
        '''
        def policy(observation):
            rand = np.random.rand()
            if rand <= epsilon:
                action = self.env.action_space.sample()
            else:
                q_values = self.estimator.get_q_value(observation)
                action = np.argmax(q_values)
            return int(action)
        return policy

    def after_train(self, reward_list):
        '''
        Method executing standart plots after training
        '''
        learning.plotting.plot_reward(reward_list)
        

    @abstractmethod
    def learn(self, episodes, epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.05,
              lambd=0.0, gamma=0.5, alpha=0.1, render=True):
        '''
        Abstract method define the learning of the agent
        '''
        raise NotImplementedError

    @abstractmethod
    def update(self, d_state, action, reward, d_next_state, next_action):
        '''
        Abstract method define the Q update
        '''
        raise NotImplementedError


class Sarsa_discreet(Simple_agent):
    '''
    Sarsa Lambda reinforcement learning algorithem (according to Barto and Suttons Reinforcement learning an intorduction).
    '''
    def __init__(self, env, logging_level=logging.INFO):
        super(Sarsa_discreet, self).__init__(env, logfile_name='Sarsa.log', logging_level=logging_level)

    def learn(self, episodes, epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.05,
              gamma=0.5, alpha=0.1, render=True):
        '''
        SARSA : reinforcement learning algorithem
        @param episodes : number of episodes we want to train
        @param epsilon : epsilon value for epsilon greedy policy
        @param epsilon_decay : decay value of epsilon for each episode
        @param epsilon_min : min epsilon value
        @param gamma : gamma value regulates the influence of the following state
        @param alpha : learning rate
        @param render : boolean value if we want to render the enviroment
        '''

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.alpha = alpha
        self.gamma = gamma

        # init plot values
        self.rewards = []
        self.solved = False
        # number = 0
        # record = {}

        # iterate over episodes
        for episode in range(episodes):
            if self.solved is True:
                print(episode)
                break

            # if self.solved is True:
            #     record.append(episode)
            #     number = number + 1
            #     if number >= 10 and sorted(record[-10:]) == record[-10:] :
            #         print(record)
            #         print('Success! You solved the mars'+
            #               'landing problem within {} '.format(episode))
            #         break

            # init enviroment
            state, reward, done, _ = self.env.reset()
            episode_reward = 0
            episode_reward += reward

            # calculate episode epsilon
            if not (self.epsilon <= self.epsilon_min):
                self.epsilon *= self.epsilon_decay

            policy = self.get_epsilon_greedy_policy(self.epsilon)  # TODO
            # get action
            action = policy(state)

            # iterate over steps
            while True:
                # do step
                next_state, reward, done, self.solved = self.env.step(action)
                reward = self.shape_reward(reward)
                next_action = policy(next_state)

                # add reward
                episode_reward += reward

                self.update(state, action, reward, next_state, next_action)

                # render the env if required
                if render:
                    still_open = self.env.render()
                    if still_open is False:
                        break

                # break condition
                if done is True:
                    self.rewards.append(episode_reward)
                    break

                action = next_action
                state = next_state
            print("\rEpisode {}/{}.  Actual Reward ({:.3f})".format(episode+1, episodes, episode_reward), end="")

        # when we are done we may want to plot a few things
        self.after_train(self.rewards)

    def shape_reward(self, reward):
        '''
        Method that handels reward shaping
        '''
        if reward != 0:
            reward = reward - 40
        return reward

    def update(self, state, action, reward, next_state, next_action=None):
        '''
        Update Q table according to following formula:
        Q(s,a) = Q(s,a) + alpha [reward + gamma * Q(next_s, next_a) - Q(s,a)]
        @param state : Actual state of the agent
        @param action : action the agent takes at state 'state'
        @param reward : reward the agent gets for taking action 'action' in state 'state'
        @param next_state : state of the agent after taking action
        @param next_action : action to take in state 'next_state'
        '''
        Q_state = self.estimator.get_q_value(state, action)
        Q_next = self.estimator.get_q_value(next_state, action)

        update = Q_state + self.alpha * (reward + self.gamma * Q_next - Q_state)

        self.estimator.update_q_value(state, action, update)


class Sarsa_lambda_discreet(Simple_agent):
    '''
    Sarsa Lambda reinforcement learning algorithem (according to Barto and Suttons Reinforcement learning an intorduction).
    '''
    def __init__(self, env, logging_level=logging.INFO):
        super(Sarsa_lambda_discreet, self).__init__(env, logfile_name='Sarsalambda.log', logging_level=logging_level)

    def learn(self, episodes, epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.05,
              gamma=0.5, alpha=0.1, lambd=0.5, render=True):
        '''
        SARSA lambda: reinforcement learning algorithem
        @param episodes : number of episodes we want to train
        @param epsilon : epsilon value for epsilon greedy policy
        @param epsilon_decay : decay value of epsilon for each episode
        @param epsilon_min : min epsilon value
        @param gamma : gamma value regulates the influence of the following state
        @param alpha : learning rate
        @param render : boolean value if we want to render the enviroment
        @param lambd : lambda value defines influence of visited states
        '''
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        # init plot values
        self.rewards = []
        self.solved = False
        # number = 0
        # record = []

        # iterate over episodes
        for episode in range(episodes):
            if self.solved is True:
                print(episode)
                break

            # if self.solved is True:
            #     record.append(episode)
            #     number = number + 1
            #     if number >= 2:
            #         if record[-1] != record[-2] + 1:
            #             number = 0
            #
            #     elif number >= 10:
            #         print(record)
            #         print('Success! You solved the mars'+
            #               'landing problem within {} '.format(episode)+'episodes')
            #         break

            # init enviroment
            state, reward, done, self.solved = self.env.reset()
            episode_reward = 0
            episode_reward += reward

            # calculate episode epsilon
            if not (self.epsilon <= self.epsilon_min):
                self.epsilon *= self.epsilon_decay

            policy = self.get_epsilon_greedy_policy(self.epsilon)  # TODO
            # get action
            action = policy(state)

            # set eglibility trace to zero TODO
            self.estimator.eg_trace = []

            # iterate over steps
            while True:
                # do step
                next_state, reward, done, self.solved = self.env.step(action)
                reward = self.shape_reward(reward)
                next_action = policy(next_state)

                # add reward
                episode_reward += reward

                # self.estimator.update_eglibility_trace(state=state, action=action, value=1)
                self.estimator.eg_trace.append([state, action, 1])

                self.update(state, action, reward, next_state, next_action)

                # render the env if required
                if render:
                    still_open = self.env.render()
                    if still_open is False:
                        break

                # brake condition
                if done is True:
                    self.rewards.append(episode_reward)
                    break

                action = next_action
                state = next_state
            print("\rEpisode {}/{}.  Actual Reward ({:.3f})".format(episode+1, episodes, episode_reward), end="")

        # when we are done we may want to plot a few things
        self.after_train(self.rewards)

    def shape_reward(self, reward):
        if reward != 0:
            reward = reward - 40
        return reward

    def update(self, state, action, reward, next_state, next_action=None):
        Q_state = self.estimator.get_q_value(state, action)
        Q_next = self.estimator.get_q_value(next_state, action)

        update = Q_state + (reward + self.gamma * Q_next - Q_state)

        for eg in self.estimator.eg_trace:
            value = self.alpha * update * eg[2]
            self.estimator.update_q_value(state=eg[0], action=eg[1], value=value)
            eg[2] *= self.lambd
