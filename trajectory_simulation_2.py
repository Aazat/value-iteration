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
        self.initial_state = [0,4]

    def get_arrival_time(self):
        return np.random.exponential(1 / self.arrival_rate)    
    
    def uniform_action_policy(self, state=None, params=None):
        """
        HOLD -> 0
        SERVE PRIORITY -> 1
        SERVE NON PRIORITY -> 2 
        """
        return random.choice([0,1,2])

    def policy_hold(self, initial_state):
        if initial_state[0]:
            return 1
        if initial_state[1] == self.Q:
            return 2
        return 0

    def policy_release(self, initial_state):
        if initial_state[0]:
            return 1        
        # if initial_state == self.Q:
        #     return 2
        return 2
    
    def expectation(self, costs):
        if not len(costs):
            return 0
        return np.mean(costs)

    def sample_next_state(self, state, policy, params):

        if not state[0]:
            params['cycle'] += 1
            params['c0'].append(self.expectation(params['c0_'])) 
            params['c1'].append(self.expectation(params['c1_'])) 
            params['gamma_1'].append(self.expectation(params['gamma_1_']))  
            params['gamma_2'].append(self.expectation(params['gamma_2_']))
            params['c0_'] = []
            params['c1_'] = []
            params['gamma_1_'] = []
            params['gamma_2_'] = []
            if params['state_counts'].get(tuple(state)):
                params['state_counts'][tuple(state)] += 1
            else:
                params['state_counts'][tuple(state)] = 1
            # print(f"State after cycle: {(params['priority'], initial_state)}")
        
        action = policy(state)

        initial_state = state        
        priority, non_priority = state[0] , state[1]
        if action == 0 and priority + non_priority == self.Q:
            return initial_state, 1e12, 1,  params
        
        if action == 0:
            # priority arrival with probability p, non-priority with 1-p (if capacity)            
            # arrival_time = np.random.exponential(1 / self.arrival_rate)
            arrival_time = 1 / self.arrival_rate
            cost_fixed = (self.W1*priority + self.W2*non_priority) * arrival_time
            params['c0_'].append(self.W2*non_priority*arrival_time)
            params['gamma_1_'].append(self.W1*priority*arrival_time)
            # print((self.W1*priority + self.W2*initial_state) * arrival_time)
            prob = 1
            if random.random() < self.p:
                # priority arrival
                priority += 1
                # cost += (self.W1 * arrival_time / 2)
                prob = self.p
            else:
                non_priority += 1
                # cost += (self.W2 * arrival_time / 2)
                prob = 1- self.p
            return [priority,non_priority], cost_fixed, prob, params
        
        if action == 1 and priority == 0:
            return initial_state, 1e12, 1, params
        
        if action == 2 and non_priority == 0:
            return initial_state, 1e12, 1, params
        
        elif action == 1:
            # arrival_time = np.random.exponential(1 / self.arrival_rate)
            # arrival_time = arrival_time if arrival_time < self.Tp else self.Tp
            # arrival_time=1 / self.arrival_rate
            priority -= 1
            if random.random() < self.P1:
                # Probability of Arrivals
                prob = self.P1
                if random.random() < self.p:
                    # priority arrival
                    priority_waiting_cost = self.W1*priority*self.Tp + (self.W1 * (self.Tp - (((1-np.exp(-self.arrival_rate*self.Tp))/(self.arrival_rate))-self.Tp*np.exp(-self.arrival_rate*self.Tp))) )
                    non_priority_waiting_cost =  self.W2*non_priority * self.Tp 
                    cost_fixed = priority_waiting_cost + non_priority_waiting_cost
                    
                    params['c1_'].append(non_priority_waiting_cost)
                    params['gamma_2_'].append(priority_waiting_cost)
                    
                    priority += 1
                    # print((self.W1*(priority-1) + self.W2*initial_state) * self.Tp,self.W1 * (self.Tp-arrival_time) / 2)
                    prob *= self.p
                    return [priority, non_priority], cost_fixed, prob, params
                
                # non priority arrival
                priority_waiting_cost = self.W1*(priority)* self.Tp
                non_priority_waiting_cost = self.W2*non_priority * self.Tp + (self.W2 * (self.Tp - (((1-np.exp(-self.arrival_rate*self.Tp))/(self.arrival_rate))-self.Tp*np.exp(-self.arrival_rate*self.Tp))))
                cost_fixed = priority_waiting_cost + non_priority_waiting_cost
                params['c1_'].append(non_priority_waiting_cost)
                params['gamma_2_'].append(priority_waiting_cost)
                non_priority += 1
                                
                # print((self.W1*(priority-1) + self.W2*initial_state) * self.Tp,self.W1 * (self.Tp-arrival_time) / 2)
                prob *= (1-self.p)
                return [priority, non_priority], cost_fixed, prob, params
            
            else:
                # No arrival
                prob = 1 - self.P1
                priority_waiting_cost = self.W1*(priority)* self.Tp
                non_priority_waiting_cost =  self.W2*non_priority* self.Tp
                cost = priority_waiting_cost + non_priority_waiting_cost
                params['c1_'].append(non_priority_waiting_cost)
                params['gamma_2_'].append(priority_waiting_cost)

                return [priority, non_priority], cost, prob, params

        elif action == 2:
            # arrival_time=1 / self.arrival_rate
            
            non_priority = non_priority - 1
            if random.random() < self.P2:
                # Probability of Arrivals
                prob = self.P2
                if random.random() < self.p:
                    # priority arrival
                    priority_waiting_cost = self.W1*priority*self.Tnp + (self.W1 * (self.Tnp) - (((1-np.exp(-self.arrival_rate*self.Tnp))/(self.arrival_rate))-self.Tnp*np.exp(-self.arrival_rate*self.Tnp)))
                    non_priority_waiting_cost =  self.W2*non_priority * self.Tnp
                    cost_fixed = priority_waiting_cost + non_priority_waiting_cost
                    params['c1_'].append(non_priority_waiting_cost)
                    params['gamma_2_'].append(priority_waiting_cost)
                    priority += 1                    
                    # print((self.W1*priority + self.W2*next_state) * self.Tnp,self.W1 * (self.Tnp-arrival_time) - arrival_time)
                    prob *= self.p
                    return [priority, non_priority], cost_fixed, prob, params
                
                # non priority arrival
                priority_waiting_cost = self.W1*priority * self.Tnp
                non_priority_waiting_cost = (self.W2*non_priority) * self.Tnp + (self.W2 * (self.Tnp) - (((1-np.exp(-self.arrival_rate*self.Tnp))/(self.arrival_rate))-self.Tnp*np.exp(-self.arrival_rate*self.Tnp)))
                cost_fixed = priority_waiting_cost + non_priority_waiting_cost
                params['c1_'].append(non_priority_waiting_cost)
                params['gamma_2_'].append(priority_waiting_cost)
                non_priority += 1                                
                # print((self.W1*priority + self.W2*next_state) * self.Tnp,self.W1 * (self.Tnp-arrival_time) - arrival_time)
                prob *= (1-self.p)
                return [priority, non_priority], cost_fixed, prob, params
            
            else:
                # No arrival
                prob = 1 - self.P2
                priority_waiting_cost = self.W1*priority* self.Tnp
                non_priority_waiting_cost =  (self.W2*non_priority) * self.Tnp
                cost = priority_waiting_cost + non_priority_waiting_cost
                params['c1_'].append(non_priority_waiting_cost)
                params['gamma_2_'].append(priority_waiting_cost)
                return [priority, non_priority], cost, prob,  params
    
    def run_trajectory(self, num_cycles=1000, initial_state = None, policy = None):
        if initial_state is None:
            initial_state = self.initial_state
        
        if policy is None:
            policy = self.uniform_action_policy
        
        current_state = initial_state
        params = {            
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
            next_state, reward, _, params = self.sample_next_state(current_state, policy, params)
            current_state = next_state
            # params['total_reward'] += reward
        params['c0_avg'] = np.mean(params['c0'])
        params['c1_avg'] = np.mean(params['c1'])
        params['gamma1_avg'] = np.mean(params['gamma_1'])
        params['gamma2_avg'] = np.mean(params['gamma_2'])
        return params
    
    def simulate_all_states(self):
        params_list = []
        states = [[i, j] for i in range(self.Q + 1) for j in range(self.Q - i + 1)]
        for initial_state in states:
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

simulator = Simulator(Q=10, p = 0.9)
# r = simulator.run_trajectory(policy = simulator.policy_hold, num_cycles=100)

# print_params(r)

params_list = simulator.simulate_all_states()

filename = "output.csv"
write_to_csv(params_list, filename)
