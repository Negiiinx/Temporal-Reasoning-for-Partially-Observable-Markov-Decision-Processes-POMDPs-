import numpy as np
import os

def retrieve_key(mapping, target_value):
    for k, v in mapping.items():
        if v == target_value:
            return k
    return None  

def standardize_array(array):
    if isinstance(array, np.ndarray):
        if array.ndim == 1:  # Vector
            total = np.sum(array)
            return array / total
        elif array.ndim == 2:  # 2D Matrix
            row_totals = np.sum(array, axis=1, keepdims=True)
            return array / row_totals
        elif array.ndim == 3:  # 3D Matrix
            row_totals = np.sum(array, axis=(1), keepdims=True)
            return array / row_totals
        else:
            raise ValueError("Array must be 1D, 2D, or 3D numpy array.")
    else:
        raise ValueError("Array must be a numpy array.")
        
def find_maximum(arr):
    max_val = np.max(arr)
    max_index = np.argmax(arr)
    return max_val, max_index

def process_states():
    states_file = "state_weights.txt"  
    with open(states_file, "r") as f:
        content = f.readlines() 
    
    state_map = {}
    value_list = []

    for line in content[2:]:
        state, val = (part.strip('"') for part in line.strip().split())
        val = int(val)
        
        if state not in state_map:
            index = len(state_map)
            state_map[state] = index
            value_list.append(val)
        else:
            index = state_map[state]
            value_list[index] += val
            
    value_list = np.array(value_list) 
    value_list = standardize_array(value_list)
    
    return state_map, value_list


def process_state_observations(states_map):
    
    obs_file = "state_observation_weights.txt"
    with open(obs_file, "r") as f:
        content = f.readlines()

    num_rows, num_states, num_obs, default_val = map(int, content[1].split())

    obs_indices = {}
    current_obs_index = 0

    weights_matrix = np.full((num_states, num_obs), default_val, dtype=int)

    for line in content[2:]:
        state, observation, val = (part.strip('"') for part in line.strip().split())
        val = int(val)

        if observation not in obs_indices:
            obs_indices[observation] = current_obs_index
            current_obs_index += 1

        state_idx = states_map[state]
        obs_idx = obs_indices[observation]
        weights_matrix[state_idx, obs_idx] = val
        
    weights_matrix = standardize_array(weights_matrix)    
    return weights_matrix, obs_indices


def process_state_actions(states_map):
    action_weights = None
    actions_map = {}
    current_action_index = 0
    actions_file = "state_action_state_weights.txt" 
    with open(actions_file, "r") as f:
        content = f.readlines()

    num_rows, num_states, num_actions, default_val = map(int, content[1].split())

    action_weights = np.full((num_states, num_states, num_actions), default_val, dtype=int)

    for line in content[2:]:
        state1, action, state2, val = (part.strip('"') for part in line.strip().split())
        val = int(val)

        if action not in actions_map:
            actions_map[action] = current_action_index
            current_action_index += 1

        state1_idx = states_map[state1]
        state2_idx = states_map[state2]
        action_idx = actions_map[action]
        action_weights[state1_idx, state2_idx, action_idx] = val
    action_weights = standardize_array(action_weights)    

    return actions_map, action_weights


def process_state_actionsdl(states_map):
    action_weights = None
    actions_map = {}
    current_action_index = 0
    actions_file = "state_action_state_weights.txt" 
    with open(actions_file, "r") as f:
        content = f.readlines()

    num_rows, num_states, num_actions, default_val = map(int, content[1].split())

    action_weights = np.full((num_states, num_states), default_val, dtype=int)

    for line in content[2:]:
        state1, action, state2, val = (part.strip('"') for part in line.strip().split())
        val = int(val)
        state1_idx = states_map[state1]
        state2_idx = states_map[state2]
        action_weights[state1_idx, state2_idx] = val
    action_weights = standardize_array(action_weights)    

    return actions_map, action_weights


def viterbi_path_finding(states_map, obs_indices, action_map, weight_matrix, action_weight_matrix, initial_probs, small_state_flag):
    obs_actions_file = "observation_actions.txt"  
    with open(obs_actions_file, "r") as f:
        content = f.readlines() 

    steps = int(content[1])
    path_probs = np.zeros((num_states, steps))
    path_trace = np.zeros((num_states, steps), dtype=int)

    tokens = content[2].strip().split()
    first_obs = tokens[0].strip('"')

    for state in states_map:
        path_probs[states_map[state]][0] = weight_matrix[states_map[state]][obs_indices[first_obs]] * initial_probs[states_map[state]]

    for i in range(1, steps):
        tokens = content[2 + i - 1].strip().split()
        current_action = tokens[1].strip('"')
        tokens = content[3 + i - 1].strip().split()
        current_obs = tokens[0].strip('"')

        optimal_prev_state = -1
        
        for state in states_map:
            highest_prob = -1
            for prev_state in states_map:
                if small_state_flag:
                    prob = path_probs[states_map[prev_state]][i - 1] * action_weight_matrix[states_map[prev_state]][states_map[state]][action_map[current_action]]
                else:
                    prob = path_probs[states_map[prev_state]][i - 1] * action_weight_matrix[states_map[prev_state]][states_map[state]]
                if prob > highest_prob:
                    highest_prob = prob
                    optimal_prev_state = states_map[prev_state]
            path_probs[states_map[state]][i] = highest_prob * weight_matrix[states_map[state]][obs_indices[current_obs]]
            path_trace[states_map[state]][i] = optimal_prev_state

    optimal_path = np.zeros(steps, dtype=int)
    highest_prob = -1
    for state in states_map:
        if path_probs[states_map[state]][steps - 1] > highest_prob:
            highest_prob = path_probs[states_map[state]][steps - 1]
            optimal_path[steps - 1] = states_map[state]
    for i in range(steps - 2, -1, -1):
        optimal_path[i] = path_trace[optimal_path[i + 1]][i + 1]

    return optimal_path, steps




def save_states(steps, path, states_map):
    output_states_file = "states.txt" 
    with open(output_states_file, "w") as output_file:
        output_file.write("states\n")
        output_file.write(str(steps) + "\n")
        for item in path:
            key = retrieve_key(states_map, item)
            output_file.write(f'"{key}"\n')

states_file = "state_weights.txt"  
with open(states_file, "r") as f:
    content = f.readlines()  
num_states = content[1].strip().split() 
num_states = int(num_states[0])


small_state_flag = 0
if num_states < 50:
    small_state_flag = 1
    
states_map, initial_probs = process_states()     
weight_matrix, obs_indices = process_state_observations(states_map)    
if small_state_flag == 0:
    _, action_weight_matrix = process_state_actionsdl(states_map)
    action_map = {}
else:
    action_map, action_weight_matrix = process_state_actions(states_map)
optimal_path, steps = viterbi_path_finding(states_map, obs_indices, action_map, weight_matrix, action_weight_matrix, initial_probs, small_state_flag)
print("Optimal Path:", optimal_path)
save_states(steps, optimal_path, states_map)
