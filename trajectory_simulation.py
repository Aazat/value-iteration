import csv
import json
import random
import numpy as np

class Simulator:
    
    def __init__(self, Q = 10, Tp = 3, Tnp = 7, p = 0.2, P1 = 0.3, P2 = 0.5, W1 = 5, W2 = 3, initial_state = 4):
        """
        STATE: Number of non-priority customer/trains in the queue.
        ACTIONS : Hold and Release
        """
        self.Q = Q
        self.Tp = Tp
        self.Tnp = Tnp
        self.p = p
        self.P1 = P1
        self.P2 = P2
        self.W1 = W1
        self.W2 = W2
        self.arrival_rate = 0.3               
        self.initial_state = initial_state

    def get_arrival_time(self):
        return np.random.exponential(1 / self.arrival_rate)    
    
    def uniform_action_policy(self, state=None, params=None):
        """
        HOLD -> 0
        SERVE PRIORITY -> 1
        SERVE NON PRIORITY -> 2 
        """
        return random.choice([0,1,2])

    def policy_hold(self, initial_state, params):
        if params['priority']:
            return 1        
        if initial_state == self.Q:
            return 2
        return 0

    def policy_release(self, initial_state, params ):
        if params['priority']:
            return 1        
        # if initial_state == self.Q:
        #     return 2
        return 2
    
    def expectation(self, costs):
        if not len(costs):
            return 0
        return np.mean(costs)

    def sample_next_state(self,initial_state, policy, params):
        # if there are no priority trains, consider start of cycle.
        if not params['priority']:
            params['cycle'] += 1
            params['c0'].append(self.expectation(params['c0_'])) 
            params['c1'].append(self.expectation(params['c1_'])) 
            params['gamma_1'].append(self.expectation(params['gamma_1_']))  
            params['gamma_2'].append(self.expectation(params['gamma_2_']))
            params['c0_'] = []
            params['c1_'] = []
            params['gamma_1_'] = []
            params['gamma_2_'] = []
            if params['state_counts'].get(initial_state):
                params['state_counts'][initial_state] += 1
            else:
                params['state_counts'][initial_state] = 1
            # print(f"State after cycle: {(params['priority'], initial_state)}")

        action = policy(initial_state, params)

        next_state = initial_state
        if action == 0 and params['priority'] + initial_state == self.Q:
            return initial_state, -1e6, params
        
        if action == 0:
            # priority arrival with probability p, non-priority with 1-p (if capacity)
            arrival_time = self.get_arrival_time()
            cost = (self.W1*params['priority'] + self.W2*initial_state) * arrival_time            
            params['c0_'].append(self.W2*initial_state*arrival_time)
            params['gamma_1_'].append(self.W1*params['priority']*arrival_time)

            # TEST_FLAG = False
            if random.random() < self.p:
                # priority arrival
                params['priority'] += 1
                params['gamma_1_'][-1] += (self.W1*arrival_time / 2)
                # print("priority_arrival", params['priority'])
                # TEST_FLAG = True
                cost += (self.W1 * arrival_time / 2)
            else:                                
                next_state += 1
                params['c0_'][-1] += (self.W2*arrival_time / 2)
                cost += (self.W2 * arrival_time / 2)
            
            # if TEST_FLAG:
            #     # print('returned params: ', params['priority'])
            return next_state, cost, params
        
        if action == 1 and params['priority'] == 0:
            return initial_state, -1e6, params
        
        if action == 2 and initial_state == 0:
            return initial_state, -1e6, params
        
        service_time = self.Tp if action == 1 else self.Tnp
        cost = (self.W1*params['priority'] + self.W2*initial_state) * service_time                

        params['priority'] -= (action == 1)
        next_state -= (action == 2)

        params['c1_'].append(self.W2*initial_state*service_time )
        params['gamma_2_'].append(self.W1*params['priority']*service_time)

        if random.random() < self.P1:
                # Probability of Arrivals
            if random.random() < self.p:
                # priority arrival
                params['priority'] += 1
                params['gamma_2_'][-1] += (self.W1 * service_time / 2)
                # print("priority arrival")
                cost += (self.W1 * service_time / 2)
            else:
                # non priority arrival
                next_state += 1
                params['c1_'][-1] += (self.W2 * service_time / 2)
                # print("non priority arrival")
                cost += (self.W2 * service_time / 2)                        
                                    
            return next_state, cost, params
        
        else:
            # No arrival
            return next_state, cost, params
    
    def run_trajectory(self, num_cycles=1000, initial_state = None, policy = None):
        if initial_state is None:
            initial_state = self.initial_state
        
        if policy is None:
            policy = self.uniform_action_policy
        
        current_state = initial_state
        params = {
            'priority' : 0,
            'cycle' : 0,
            'c0_' : [], 'c0' : [], 
            'c1_' : [], 'c1' : [], 
            'gamma_1_' : [], 'gamma_1' : [], 
            'gamma_2_' : [],'gamma_2' : [],
            # 'total_reward' : 0,
            'state_counts' : {}
        }
        total_reward = 0
        while params['cycle'] < num_cycles:
            next_state, reward, params = self.sample_next_state(current_state, policy, params)
            current_state = next_state
            # params['total_reward'] += reward
        params['c0_avg'] = np.mean(params['c0'])
        params['c1_avg'] = np.mean(params['c1'])
        params['gamma1_avg'] = np.mean(params['gamma_1'])
        params['gamma2_avg'] = np.mean(params['gamma_2'])
        return params
    
    def simulate_all_states(self):
        params_list = []
        for initial_state in range(self.Q):
            params_hold = self.run_trajectory(initial_state = initial_state, policy = self.policy_hold)
            params_release = self.run_trajectory(initial_state = initial_state, policy = self.policy_release)
            params_list.append({
                'params_hold' : params_hold,
                'params_release' : params_release,                
            })
        return params_list

def print_params(params):
    print(f"Number of Cycles: {params['cycle']}\nState Counts: {params['state_counts']}")

# Function to write data to CSV
def write_to_csv(data, filename):
    # Define CSV column headers
    headers = ['cycle', 'action', 'c_0', 'c_1', 'gamma_1', 'gamma_2']
    
    # Open the file for writing
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header
        csvwriter.writerow(headers)
        
        # Write the rows
        cycle = 1
        for item in data:
            # Write hold action row
            hold_values = item['params_hold']
            csvwriter.writerow([cycle, 'hold', hold_values['c0_avg'], hold_values['c1_avg'], hold_values['gamma1_avg'], hold_values['gamma2_avg']])
            
            # Write release action row
            release_values = item['params_release']
            csvwriter.writerow([cycle, 'release', release_values['c0_avg'], release_values['c1_avg'], release_values['gamma1_avg'], release_values['gamma2_avg']])
            
            # Increment cycle
            cycle += 1

simulator = Simulator(Q=20, p = 0.9)
r = simulator.run_trajectory(policy = simulator.policy_hold, num_cycles=100)

# print_params(r)

params_list = simulator.simulate_all_states()

filename = "output.csv"
write_to_csv(params_list, filename)
