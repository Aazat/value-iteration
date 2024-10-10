import random
import numpy as np 
import pandas as pd

class PolicyIteration:
    def __init__(self, Q = 100, Tp = 3, Tnp = 7, p = 0.2, P1 = 0.3, P2 = 0.5, W1 = 5, W2 = 3, arrival_rate = 0.4 ):
        self.Q = Q
        self.Tp = Tp
        self.Tnp = Tnp
        self.p = p
        self.P1 = P1
        self.P2 = P2
        self.W1 = W1
        self.W2 = W2
        self.arrival_rate = arrival_rate

    def sample_next_state(self, state, action):
        # if there are no priority trains, consider start of cycle.    
        
        params = {'priority' : state[0]}
        initial_state = state[1]

        next_state = state[1]
        if action == 0 and params['priority'] + initial_state == self.Q:
            return initial_state, 100, params, 1
        
        if action == 0:
            # priority arrival with probability p, non-priority with 1-p (if capacity)
            # arrival_time = get_arrival_time()
            arrival_time = 1 / self.arrival_rate
            cost = (self.W1*params['priority'] + self.W2*initial_state) * arrival_time                    
            prob = 1
            if random.random() < self.p:
                # priority arrival
                params['priority'] += 1
                cost += (self.W1 * arrival_time / 2)
                prob = self.p
            else:
                next_state += 1
                cost += (self.W2 * arrival_time / 2)
                prob = 1- self.p
            return next_state, cost, params, prob
        
        if action == 1 and params['priority'] == 0:
            return initial_state, 1e6, params, 1
        
        if action == 2 and initial_state == 0:
            return initial_state, 1e6, params, 1
        
        service_time = self.Tp if action == 1 else self.Tnp
        cost = (self.W1*params['priority'] + self.W2*initial_state) * service_time        

        params['priority'] -= (action == 1)
        next_state -= (action == 2)

        if random.random() < self.P1:
            # Probability of Arrivals
            prob = self.P1
            if random.random() < self.p:
                # priority arrival
                params['priority'] += 1
                cost += (self.W1 * service_time / 2)
                prob *= self.p
            else:
                # non priority arrival
                next_state += 1
                cost += (self.W2 * service_time / 2)
                prob *= (1-self.p)
            return next_state, cost, params, prob
        
        else:
            # No arrival
            prob = 1 - self.P1
            return next_state, cost, params, prob
    
    def get_transition_set(self, state, action):    
        tset = set()
        for _ in range(1000):
            non_priority, cost,params,prob = self.sample_next_state(state, action)
            next_state = (params['priority'], non_priority)
            tset.add(((next_state), prob, cost))
        return tset

    def value_iteration(self, gamma = 0.9, theta = -1e6, max_iterations = 1000):
        states = [[i,j] for i in range(self.Q+1) for j in range(self.Q-i + 1)]
        transition_reward_matrix = {}
        actions = [0,1,2]        

        for state in states:
            transition_reward_matrix[tuple(state)] = {}
            for action in actions:
                tset = self.get_transition_set(state,action)
                transition_reward_matrix[tuple(state)][action] = tset    
        
        value_function = {}
        for state in transition_reward_matrix:
            value_function[tuple(state)] = 0
        
        policy = {}

        delta = float('inf')
        iteration = 0
        while delta > theta and iteration < max_iterations:
            iteration += 1
            delta = 0
            for state, action_dict in transition_reward_matrix.items():
                old_value = value_function[state]
                action_values = []
                for action,tset in action_dict.items():
                    action_value = 0
                    for transition in tset:
                        next_state = transition[0]
                        prob = transition[1]
                        cost = transition[2]
                        action_value += prob * (-1 * cost + gamma * value_function[next_state])
                    action_values.append(action_value)

                value_function[state] = max(action_values)
                policy[state] = np.argmax(action_values)
                delta = max(delta, abs(value_function[state] - old_value))
        
        return value_function, policy
    
    def policy_evaluation(self, policy, gamma=0.9, theta=1e-6, max_iterations=1000):
        states = [[i, j] for i in range(self.Q+1) for j in range(self.Q-i + 1)]
        value_function = {tuple(state): 0 for state in states}

        for _ in range(max_iterations):
            delta = 0
            for state in states:
                state_tuple = tuple(state)
                old_value = value_function[state_tuple]
                action = policy[state_tuple]
                action_value = 0
                tset = self.get_transition_set(state, action)
                for transition in tset:
                    next_state = transition[0]
                    prob = transition[1]
                    cost = transition[2]
                    action_value += prob * (-1 * cost + gamma * value_function[tuple(next_state)])
                
                value_function[state_tuple] = action_value
                delta = max(delta, abs(old_value - action_value))
            
            if delta < theta:
                break
        
        return value_function

    def policy_iteration(self, gamma=0.9, max_iterations=1000):
        states = [[i, j] for i in range(self.Q+1) for j in range(self.Q-i + 1)]
        actions = [0, 1, 2]
        
        policy = {tuple(state): random.choice(actions) for state in states}
        stable_policy = False
        
        while not stable_policy:
            stable_policy = True
            value_function = self.policy_evaluation(policy, gamma=gamma)
            
            for state in states:
                state_tuple = tuple(state)
                old_action = policy[state_tuple]
                
                action_values = []
                for action in actions:
                    action_value = 0
                    tset = self.get_transition_set(state, action)
                    for transition in tset:
                        next_state = transition[0]
                        prob = transition[1]
                        cost = transition[2]
                        action_value += prob * (-1 * cost + gamma * value_function[tuple(next_state)])
                    action_values.append(action_value)
                
                best_action = np.argmax(action_values)
                policy[state_tuple] = best_action
                
                if old_action != best_action:
                    stable_policy = False
        
        return value_function, policy

def output(value_function, policy):
    output_df = pd.DataFrame(list(policy.items()), columns=["state", "optimal_action"])
    output_df["state_value"] = list(value_function.values())
    output_df.to_csv("state_value_action2.csv", index=False)

# value_function, policy = ValueIteration().value_iteration()
# output(value_function=value_function, policy=policy)
    
# Running the policy iteration
policy_iteration_instance = PolicyIteration()
value_function, policy = policy_iteration_instance.policy_iteration()

# Output results
output(value_function=value_function, policy=policy)
