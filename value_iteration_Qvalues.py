import random
import numpy as np
import pandas as pd
from itertools import product
import os

class ValueIteration:
    def __init__(self, Q=100, Tp=3, Tnp=7, p=0.2, P1=0.3, P2=0.5, W1=5, W2=3, arrival_rate=0.4):
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
        params = {'priority': state[0]}
        initial_state = state[1]
        next_state = state[1]

        if action == 0 and params['priority'] + initial_state == self.Q:
            return initial_state, 1e6, params, 1

        if action == 0:
            arrival_time = 1 / self.arrival_rate
            cost = (self.W1 * params['priority'] + self.W2 * initial_state) * arrival_time
            prob = 1
            if random.random() < self.p:
                params['priority'] += 1
                cost += (self.W1 * arrival_time / 2)
                prob = self.p
            else:
                next_state += 1
                cost += (self.W2 * arrival_time / 2)
                prob = 1 - self.p
            return next_state, cost, params, prob

        if action == 1 and params['priority'] == 0:
            return initial_state, 1e6, params, 1

        if action == 2 and initial_state == 0:
            return initial_state, 1e6, params, 1

        service_time = self.Tp if action == 1 else self.Tnp
        cost = (self.W1 * params['priority'] + self.W2 * initial_state) * service_time
        p = self.P1 if action == 1 else self.P2

        params['priority'] -= (action == 1)
        next_state -= (action == 2)

        if random.random() < p:
            prob = p
            if random.random() < self.p:
                params['priority'] += 1
                cost += (self.W1 * service_time / 2)
                prob *= self.p
            else:
                next_state += 1
                cost += (self.W2 * service_time / 2)
                prob *= (1 - self.p)
            return next_state, cost, params, prob

        else:
            prob = 1 - p
            return next_state, cost, params, prob

    def get_transition_set(self, state, action):
        tset = set()
        for _ in range(1000):
            non_priority, cost, params, prob = self.sample_next_state(state, action)
            next_state = (params['priority'], non_priority)
            tset.add(((next_state), prob, cost))
        return tset

    def value_iteration(self, gamma=0.9, theta=-1e6, max_iterations=1000):
        states = [[i, j] for i in range(self.Q+1) for j in range(self.Q-i + 1)]
        transition_reward_matrix = {}
        actions = [0, 1, 2]

        for state in states:
            transition_reward_matrix[tuple(state)] = {}
            for action in actions:
                tset = self.get_transition_set(state, action)
                transition_reward_matrix[tuple(state)][action] = tset

        value_function = {tuple(state): 0 for state in transition_reward_matrix}
        Q_values = {tuple(state): {action: 0 for action in actions} for state in transition_reward_matrix}
        policy = {}

        delta = float('inf')
        iteration = 0
        while delta > theta and iteration < max_iterations:
            iteration += 1
            delta = 0
            for state, action_dict in transition_reward_matrix.items():
                old_value = value_function[state]
                action_values = []
                for action, tset in action_dict.items():
                    action_value = 0
                    for transition in tset:
                        next_state = transition[0]
                        prob = transition[1]
                        cost = transition[2]
                        action_value += prob * (-1 * cost + gamma * value_function[next_state])
                    Q_values[state][action] = action_value
                    action_values.append(action_value)

                value_function[state] = max(action_values)
                policy[state] = np.argmax(action_values)
                delta = max(delta, abs(value_function[state] - old_value))

        return value_function, Q_values, policy, transition_reward_matrix

def export_transition_reward_matrix(transition_reward_matrix, file_prefix, folder_name):
    transition_list = []
    reward_list = []
    for state, action_dict in transition_reward_matrix.items():
        for action, tset in action_dict.items():
            for transition in tset:
                next_state, prob, cost = transition
                transition_list.append((state, action, next_state, prob, cost))
                # reward_list.append((state, action, next_state, cost))
    
    transition_df = pd.DataFrame(transition_list, columns=["state", "action", "next_state", "probability", "reward"])
    # reward_df = pd.DataFrame(reward_list, columns=["state", "action", "next_state", "reward"])

    transition_df.to_csv(f"{folder_name}/{file_prefix}_transition_matrix.csv", index=False)
    # reward_df.to_csv(f"{folder_name}/{file_prefix}_reward_matrix.csv", index=False)

def export_value_function(value_function, policy, file_prefix, folder_name):
    value_list = [(state, policy[state], value) for state, value in value_function.items()]
    value_df = pd.DataFrame(value_list, columns=["state", "optimal_action", "state_value"])
    value_df.to_csv(f"{folder_name}/{file_prefix}_value_function.csv", index=False)

def run_simulations(parameter_values, folder_name):
    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    param_names = list(parameter_values.keys())
    
    for param_combination in product(*parameter_values.values()):
        params = dict(zip(param_names, param_combination))
        vi_instance = ValueIteration(**params)
        value_function, Q_values, policy, transition_reward_matrix = vi_instance.value_iteration()

        # Generate a unique file prefix based on the parameter combination
        file_prefix = "_".join([f"{k}_{v}" for k, v in params.items()])

        # Export Q-values
        Q_list = []
        for state, actions in Q_values.items():
            for action, Q_value in actions.items():
                Q_list.append((state, action, Q_value))
        Q_df = pd.DataFrame(Q_list, columns=["state", "action", "Q_value"])
        Q_df.to_csv(f"{folder_name}/{file_prefix}_Q_values.csv", index=False)

        # Export transition and reward matrices
        export_transition_reward_matrix(transition_reward_matrix, file_prefix, folder_name)
        export_value_function(value_function, policy, file_prefix, folder_name)

# Example usage:
parameter_values = {
    "Q": [10],
    "Tp": [5],
    "Tnp": [10],
    "p": [0.3],
    "P1": [0.4],
    "P2": [0.6],
    "W1": [10],
    "W2": [6],
    "arrival_rate": [0.5]
}

# Specify the folder name
folder_name = "simulation_results"

# Run simulations and export results
run_simulations(parameter_values, folder_name)
