import numpy as np
import itertools
from collections import defaultdict
from environment import Environment
import dill as pickle

class Agent():

    def __init__(self, alpha, epsilon, gamma):
        '''
            Q(s,a) -> action value
            model(s,a) -> R, s'
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.reward_space = np.array(['-1', '1000', '-1000', '-10'])
        self.state_space = np.array(['0', '1'])
        self.Q = defaultdict(lambda: np.random.randint(5,size=5))
        self.model_reward = defaultdict(lambda: np.random.choice(self.reward_space, 5).astype(np.int))
        self.model_obs = defaultdict(lambda : np.random.choice(self.state_space, 25).astype(np.int))

    def greedy_policy(self, state):
        '''
            returns action from state using epsilon greedy policy
            derived from Q
        '''
        prob = np.random.random()
        if prob <= self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.Q[state])

    def play(self):
        
        score = np.zeros(episode_num)
        for episode in range(episode_num):
            
            #env.render()
            state = env.reset()
            action = self.greedy_policy(tuple(state))
    
            observation, reward, done = env.step(action)
            next_action = np.argmax(self.Q[tuple(observation)])
            self.Q[tuple(state)][action] += self.alpha * (reward + (self.gamma * self.Q[tuple(observation)][next_action]) - self.Q[tuple(state)][action])
            self.model_reward[tuple(state)][action] = reward
            self.model_obs[tuple(state)][action*5:(action*5)+5] = tuple(observation)
            
            for step in range(20):
                
                random_state = list(self.Q.keys())[np.random.choice(len(list(self.Q.keys())))]
                random_action = np.random.randint(5)
                R = self.model_reward[tuple(random_state)][random_action]
                print("Reward and Action ",R, random_action)
                next_S = self.model_obs[tuple(random_state)][random_action*5: (random_action*5)+5]
                a = np.argmax(self.Q[tuple(next_S)])
                self.Q[tuple(random_state)][random_action] += self.alpha * (R + (self.gamma * self.Q[tuple(next_S)][a]) - self.Q[tuple(random_state)][random_action])
                #env.render()
                if done:
                    print("Finished episode", episode+1)
                    score[episode] = step
                    break
                print("Finished time step "+str(step)+"of episode"+str(episode+1))
        return score, self.Q

env = Environment()
agent = Agent(0.1,0.1,0.95)
episode_num = 500
score, policy = agent.play()
pol = policy
print(dict(policy))
print(len(policy))
