import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product

DIR_PATH = 'relative_value_iteration_results'

if not os.path.exists("relative_value_iteration_results"):
    os.makedirs("relative_value_iteration_results")

class ValueIteration:
    def __init__(self, Q=5, Tp=3, Tnp=3, p=0.5, P1=0.3, P2=0.5, W1=5, W2=3, arrival_rate=0.2):
        self.Q = Q
        self.Tp = Tp
        self.Tnp = Tnp
        self.W1 = W1
        self.W2 = W2
        self.arrival_rate = arrival_rate
        self.p = p
        self.P1 = (1-np.exp(-Tp*arrival_rate))
        self.P2 = (1-np.exp(-Tnp*arrival_rate))
        self.action_times = {
            "0" : 1 / arrival_rate,
            "1" : Tp,
            "2" : Tnp
        }
        self.states = [(i, j) for i in range(self.Q + 1) for j in range(self.Q - i + 1)]
        if arrival_rate * Tp > 1 or arrival_rate * Tnp > 1:
            print('Arrival rate too high')
            raise AssertionError("Arrival rate too high")

    # def sample_next_state(self, state, action):
    #     params = {'priority': state[0]}
    #     initial_state = state[1]
    #     next_state = state[1]
    #     if action == 0 and params['priority'] + initial_state == self.Q:
    #         return initial_state, 1e12, params, 1, 0
        
    #     if action == 0:
    #         arrival_time = 1 / self.arrival_rate
    #         cost = (self.W1*params['priority'] + self.W2*initial_state) * arrival_time
    #         # cost = (self.W1*params['priority'] + self.W2*initial_state) * (1 / self.arrival_rate)
    #         prob = 1       
    #         if random.random() < self.p:
    #             # priority arrival
    #             params['priority'] += 1
    #             prob = self.p
    #         else:
    #             next_state += 1
    #             prob = 1- self.p
    #         return next_state, cost, params, prob, arrival_time
        
    #     if action == 1 and params['priority'] == 0:
    #         return initial_state, 1e12, params, 1, 0
        
    #     if action == 2 and initial_state == 0:
    #         return initial_state, 1e12, params, 1, 0
        
    #     elif action == 1:
    #         # arrival_time = np.random.exponential(1 / self.arrival_rate)
    #         # arrival_time = arrival_time if arrival_time < self.Tp else self.Tp
    #         # arrival_time=1 / self.arrival_rate
    #         params['priority'] -= 1
    #         if random.random() < self.P1:
    #             # Probability of Arrivals
    #             prob = self.P1
    #             if random.random() < self.p:
    #                 # priority arrival
    #                 params['priority'] += 1
    #                 # cost =(self.W1*(params['priority']-1) + self.W2*initial_state) * self.Tp+ (self.W1 * (self.Tp-arrival_time))
    #                 cost = (self.W1*(params['priority']-1) + self.W2*initial_state) * self.Tp+ (self.W1 * (self.Tp- ((1-np.exp(-self.arrival_rate*self.Tp))/(self.arrival_rate))))
    #                 prob *= self.p
    #             else:
    #                 # non priority arrival
    #                 next_state += 1
    #                 #cost = (self.W1*(params['priority']-1) + self.W2*initial_state) * self.Tp+(self.W2 * (self.Tp-arrival_time))
    #                 cost = (self.W1*(params['priority']-1) + self.W2*initial_state) * self.Tp+(self.W2 * (self.Tp-((1-np.exp(-self.arrival_rate*self.Tp))/(self.arrival_rate))))
    #                 prob *= (1-self.p)
    #             return next_state, cost, params, prob,self.Tp
            
    #         else:
    #             # No arrival
    #             prob = 1 - self.P1
    #             cost =(self.W1*(params['priority']) + self.W2*initial_state)* self.Tp
    #             return next_state, cost, params, prob,self.Tp

    #     elif action == 2:
    #         # arrival_time=1 / self.arrival_rate
    #         # arrival_time = np.random.exponential(1 / self.arrival_rate)
    #         # arrival_time = arrival_time if arrival_time < self.Tnp else self.Tnp
    #         next_state = initial_state - 1
    #         if random.random() < self.P2:
    #             # Probability of Arrivals
    #             prob = self.P2
    #             if random.random() < self.p:
    #                 # priority arrival
    #                 params['priority'] += 1
    #                 #cost =(self.W1*params['priority'] + self.W2*next_state) * self.Tnp+ (self.W1 * (self.Tnp-arrival_time))
    #                 cost =(self.W1*params['priority'] + self.W2*next_state) * self.Tnp + (self.W1 * (self.Tnp-((1-np.exp(-self.arrival_rate*self.Tnp))/(self.arrival_rate))))
    #                 prob *= self.p
    #             else:
    #                 # non priority arrival
    #                 next_state += 1
    #                 #cost = (self.W1*params['priority'] + self.W2*next_state) * self.Tnp+(self.W2 * (self.Tnp-arrival_time))
    #                 cost = (self.W1*params['priority'] + self.W2*next_state) * self.Tnp + (self.W2 * (self.Tnp-((1-np.exp(-self.arrival_rate*self.Tnp))/(self.arrival_rate))))
    #                 prob *= (1-self.p)
    #             return next_state, cost, params, prob,self.Tnp
            
    #         else:
    #             # No arrival
    #             prob = 1 - self.P2
    #             cost = (self.W1*params['priority'] + self.W2*next_state) * self.Tnp
    #             return next_state, cost, params, prob,self.Tnp    
            
    def action_0(self, state):
        if state[0] + state[1] == self.Q:
            return [(state, 1, 1e12)]
        cost = (self.W1 * state[0] + self.W2 * state[1])  / self.arrival_rate
        transitions = [(((state[0] + 1, state[1])), self.p, cost), ((state[0], state[1] + 1), 1- self.p, cost)]
        
        return transitions
    
    def action_1(self, state):
        if state[0] == 0:
            return [(state , 1, 1e12)]
        
        transitions = []
        # N.A
        cost = ((state[0] - 1)* self.W1 + state[1] * self.W2) * self.Tp
        transitions.append(((state[0] - 1, state[1]), 1 - self.P1, cost))

        # P.A
        priority_arrival_cost = self.W1 * (self.Tp- ((1-np.exp(-self.arrival_rate*self.Tp))/(self.arrival_rate)))
        transitions.append((state, self.p * self.P1, cost + priority_arrival_cost))

        # N.P.A
        non_priority_cost = self.W2 * (self.Tp-((1-np.exp(-self.arrival_rate*self.Tp))/(self.arrival_rate)))
        transitions.append(((state[0]-1, state[1] + 1), (1 - self.p) * self.P1, cost + non_priority_cost))
        return transitions
    
    def action_2(self, state):
        if state[1] == 0:
            return [(state, 1, 1e12)]
        transitions = []

        # N.A
        cost = (state[0] * self.W1 + (state[1] - 1) * self.W2) * self.Tnp
        transitions.append(((state[0], state[1] - 1), 1 - self.P2, cost))

        # P.A
        priority_arrival_cost = (self.W1 * (self.Tnp-((1-np.exp(-self.arrival_rate*self.Tnp))/(self.arrival_rate))))
        transitions.append(((state[0] + 1, state[1]-1), self.p * self.P2, cost + priority_arrival_cost))

        # N.P.A
        non_priority_cost = (self.W2 * (self.Tnp-((1-np.exp(-self.arrival_rate*self.Tnp))/(self.arrival_rate))))
        transitions.append(((state[0], state[1]), (1 - self.p) * self.P2, cost + non_priority_cost))

        return transitions
    
    def get_transition_reward_matrix(self):        

        transition_reward_matrix = {}

        for state in self.states:
            action_0_result = self.action_0(state)
            action_1_result = self.action_1(state)
            action_2_result = self.action_2(state)
            transition_reward_matrix[state] = (action_0_result, action_1_result, action_2_result)
            transition_reward_matrix[state] = {
                '0' : action_0_result,
                '1' : action_1_result,
                '2' : action_2_result
            }
        return transition_reward_matrix      

    # def get_transition_reward_matrix_from_sampling(self):
    #     states = [(i, j) for i in range(self.Q + 1) for j in range(self.Q - i + 1)]
    #     transition_reward_matrix = {}
    #     actions = [0, 1, 2]

    #     for state in states:
    #         transition_reward_matrix[tuple(state)] = {}
    #         for action in actions:
    #             tset = self.get_transition_set(state, action)
    #             transition_reward_matrix[tuple(state)][action] = tset
        
    #     return transition_reward_matrix

    # def get_transition_set(self, state, action):
    #     tset = set()
        
    #     for _ in range(1000):
    #         non_priority, cost, params, prob, time_for_discounting = self.sample_next_state(state, action)
    #         next_state = (params['priority'], non_priority)
    #         tset.add(((next_state), prob, cost, time_for_discounting))
    #     return tset

    def value_iteration(self, gamma=0.001, theta=1e-5, max_iterations=10000):
        states = [[i, j] for i in range(self.Q + 1) for j in range(self.Q - i + 1)]
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
            print(f"Iteration: {iteration}")
            delta = 0
            for state, action_dict in transition_reward_matrix.items():
                old_value = value_function[state]
                action_values = []
                for action, tset in action_dict.items():
                    action_value = 0
                    time_for_discounting = self.action_times[action]
                    for transition in tset:
                        next_state, prob, cost = transition
                        # print(state, next_state, prob)                        
                        # dummy_1, cost, dummy_2, dummy_3,time_for_discounting=self.sample_next_state(state, action)
                        action_value += (-1 * cost + np.exp(-time_for_discounting*gamma)*prob  * value_function[next_state])
                        # action_value += (-1 * cost + np.exp(-time_for_discounting*gamma)  * value_function[next_state])*prob
                    Q_values[state][action] = action_value
                    action_values.append(action_value)

                value_function[state] = max(action_values)
                policy[state] = np.argmax(action_values)
                print(f"State: {state}, Value: {value_function[state]}, Policy: {policy[state]}")
                delta = max(delta, abs(value_function[state] - old_value))

        return value_function, Q_values, policy, transition_reward_matrix
    
    def relative_value_iteration(self, theta = 1e-5, max_iterations = 10000):        
        # transition_reward_matrix = {}
        actions = [0, 1, 2]

        # for state in states:
        #     transition_reward_matrix[tuple(state)] = {}
        #     for action in actions:
        #         tset = self.get_transition_set(state, action)
        #         transition_reward_matrix[tuple(state)][action] = tset

        transition_reward_matrix = self.get_transition_reward_matrix()

        relative_value_function = {state: 0 for state in transition_reward_matrix}
        Q_values = {state: {action: 0 for action in actions} for state in transition_reward_matrix}
        policy = {}

        reference_state = self.states[0]
        delta = float('inf')
        iteration = 0
        while delta > theta and iteration < max_iterations:
            iteration += 1
            print(f"Iteration: {iteration}")
            delta = 0
            g = 0

            for state, action_dict in transition_reward_matrix.items():
                old_value = relative_value_function[state]
                action_values = []

                for action, tset in action_dict.items():
                    action_value = 0
                    time_for_discounting = self.action_times[action]
                    for transition in tset:
                        if len(transition) != 3:
                            print("problem here")
                        next_state, prob, cost = transition
                        if time_for_discounting == 0:
                            # print("zero here")
                            action_value += prob * (cost + relative_value_function[next_state])
                        else:
                            action_value += prob * (cost / time_for_discounting + relative_value_function[next_state])

                    Q_values[state][action] = action_value
                    action_values.append(action_value)

                relative_value_function[state] = min(action_values)
                policy[state] = np.argmin(action_values)

                print(f"State: {state}, Value: {relative_value_function[state]}, Policy: {policy[state]}")

                g = relative_value_function[reference_state]
                relative_value_function[state] -= g

                delta = max(delta, abs(relative_value_function[state] - old_value))

        return relative_value_function, Q_values, policy, transition_reward_matrix


def export_transition_reward_matrix(transition_reward_matrix):
    transition_list = []
    reward_list = []
    for state, action_dict in transition_reward_matrix.items():
        for action, tset in action_dict.items():
            for transition in tset:
                next_state, prob, cost = transition
                transition_list.append((state, action, next_state, prob, cost))
                # reward_list.append((state, action, next_state, cost))
    
    transition_df = pd.DataFrame(transition_list, columns=["state", "action", "next_state", "probability", "cost"])
    # reward_df = pd.DataFrame(reward_list, columns=["state", "action", "next_state", "reward"])

    transition_df.to_csv("relative_value_iteration_results/transition_matrix.csv", index=False)
    # reward_df.to_csv("reward_matrix.csv", index=False)

def run_simulation_for_params(params):
    """
    Run simulation for a given set of parameters and return the results.
    """
    vi_instance = ValueIteration(**params)
    _, Q_values, _, transition_reward_matrix = vi_instance.value_iteration()

    results = []
    for state, actions in Q_values.items():
        for action, Q_value in actions.items():
            result_row = {**params, 'state': state, 'action': action, 'Q_value': Q_value}
            results.append(result_row)
    
    # Export transition and reward matrices for this set of parameters
    export_transition_reward_matrix(transition_reward_matrix)
    
    return results

def export_value_function_policy(value_function, policy, params):
    results = []
    for state, value in value_function.items():
        result_row = {**params, 'state' : state, 'value' : value, 'optimal_action' : policy[state]}
        results.append(result_row)
    
    pd.DataFrame(results).to_csv(os.path.join(DIR_PATH, 'value_function.csv'))

def relative_value_iteration_for_params(params):
    """
    Run simulation for a given set of parameters and return the results.
    """
    vi_instance = ValueIteration(**params)
    relative_value_function, Q_values, policy, transition_reward_matrix = vi_instance.relative_value_iteration()

    results = []    
    for state, actions in Q_values.items():
        for action, Q_value in actions.items():
            result_row = {**params, 'state': state, 'action': action, 'Q_value': Q_value}
            results.append(result_row)
    
    # Export transition and reward matrices for this set of parameters
    export_transition_reward_matrix(transition_reward_matrix)
    export_value_function_policy(value_function= relative_value_function, params=params, policy= policy)
    
    return results


def run_simulations(parameter_values):
    param_names = list(parameter_values.keys())
    param_combinations = list(product(*parameter_values.values()))

    # Prepare the pool for multiprocessing
    pool = mp.Pool(20)

    # Create a list of dictionaries of parameters
    param_dicts = [dict(zip(param_names, combination)) for combination in param_combinations]

    # Step 1: Run simulations in parallel
    # all_results = pool.map(run_simulation_for_params, param_dicts)
    all_results = pool.map(relative_value_iteration_for_params, param_dicts)

    # Step 2: Flatten the list of results
    flat_results = [item for sublist in all_results for item in sublist]

    # Step 3: Create DataFrame from results list
    result_df = pd.DataFrame(flat_results)

    # Save to CSV with all parameters as separate columns
    # result_df.to_csv("Q_values_simulations.csv", index=False)
    result_df.to_csv("relative_value_iteration_results/relative_value_iterations.csv", index=False)

    pool.close()
    pool.join()

if __name__ == '__main__':
    # Example usage:
    parameter_values = {
        "Q": [5],
        "Tp": [3],
        "Tnp": [3],
        "p": [0.5],
        "P1": [0.3],
        "P2": [0.5],
        "W1": [35],
        "W2": [1],
        "arrival_rate": [0.2]
    }

    run_simulations(parameter_values)

    # Step 1: Read the Q_values_simulations.csv file into a DataFrame
    q_values_df = pd.read_csv("relative_value_iteration_results/relative_value_iterations.csv")

    # Step 2: Identify the parameter columns
    parameter_columns = ['Q', 'Tp', 'Tnp', 'p', 'P1', 'P2', 'W1', 'W2', 'arrival_rate']

    # Step 3: Filter the DataFrame to include only actions '0' and '2'
    filtered_df = q_values_df[q_values_df['action'].isin([0, 2])]

    # Step 4: Aggregate duplicate entries by taking the mean of Q_value
    # Group by parameter columns, state, and action
    aggregated_df = filtered_df.groupby(parameter_columns + ['state', 'action'], as_index=False)['Q_value'].mean()

    # Step 5: Pivot the DataFrame to have actions as columns for easier difference calculation
    pivot_df = aggregated_df.pivot(index=parameter_columns + ['state'], columns='action', values='Q_value')

    # Step 6: Calculate the difference between Q-values for actions '2' and '0'
    pivot_df['Difference'] = pivot_df[2] - pivot_df[0]

    # Step 7: Reset index to make parameter columns part of the DataFrame again
    difference_df = pivot_df[['Difference']].reset_index()

    # Step 8: Filter the DataFrame to include only rows where the state begins with '(0,'
    difference_df = difference_df[difference_df['state'].astype(str).str.startswith("(0,")]

    # Step 9: Export the new DataFrame to Difference.csv
    difference_df.to_csv("relative_value_iteration_results/Difference.csv", index=False)
