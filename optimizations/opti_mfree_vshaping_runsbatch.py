import os
import numpy as np
import json
import time
from scipy.optimize import differential_evolution

from utils.social_functions import social_sim_mf
from models.mf_valueshaping import mf_valueshaping

"""Optimization of the model-free social agent with value shaping using differential evolution."""

# Parameters
n_episodes = 20
n_simulations = 1000 # 1000
max_steps = 40 #40
n_calls = 15
training = 0.5
popsize = 5

seed = 5
rng = np.random.default_rng(seed)


print(f"MF VS diff alg w/ max_steps {max_steps}, n_episodes {n_episodes}, n_simulations {n_simulations}, n_calls {n_calls}")

# Define the parameter space
space = [
    (-5, 5),  # Unbounded space for inverse temperature Real(-5, 5)
    (-10, 10),  # Unbounded space for discount factor Real(-10, 10)
    (-5,5) # kappa - social parameter Real(-5,5)
]


## LOAD DATA ##
# Load learning parameters of the model-free asocial agent
with open('saved/opti_results/mfree_agent.json', 'r') as json_file:
    mb_agent_params = json.load(json_file)["opti_params"]
    alpha = mb_agent_params["alpha"]
# Load data from the expert
with open('saved/baseline/mbased_expert_baseline.json', 'r') as json_file:
    expert_data= json.load(json_file)
# Convert the saved lists to array
for k in expert_data.keys():
    expert_data[k] = np.array(expert_data[k])

# Load the worlds so they don't have to be created again
loaded = np.load('saved/worlds.npz')
worlds_saved = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

rewards_load = np.load('saved/rewards_info.npz')
rewards_shuffled = [rewards_load[f'arr_{i}'] for i in range(len(rewards_load.files))]


## OPTIMIZATION ##
def objective_function(unbounded_params, mf_valueshaping, expert_data,  worlds_saved, rewards_shuffled, n_simulations, max_steps, n_episodes, training, rng):

    # Bound parameters
    # Apply inverse transformations for continuous parameters
    beta = np.exp(unbounded_params[0])                    # For [0, +inf) bounded
    gamma = 1 / (1 + np.exp(-unbounded_params[1]))        # For [0, 1] bounded
    kappa_vs = np.exp(unbounded_params[2])

    params  = {"beta": beta, "alpha": alpha, "gamma": gamma, "kappa":kappa_vs}
    #print("params", params)

    rewards_result = np.zeros((n_simulations, n_episodes))
    #start_objective = time.time()

        
    rewards_result = social_sim_mf(mf_valueshaping, expert_data, worlds_saved, rewards_shuffled, n_simulations, max_steps, n_episodes, params, training, rng, optimization = True, world_model = 'baseline', rewards_exp2=None)

    #period = (time.time() - start_objective)/60
    #print("1 set of params takes", period, "seconds")

    #print("Optimizing over training", -np.mean(rewards_result[:, :int(n_episodes*training)]))
    return -np.mean(rewards_result[:, :int(n_episodes*training)]) # mean of  

#result_time = time.time()
result = differential_evolution(objective_function, space, args=(mf_valueshaping, expert_data, worlds_saved, rewards_shuffled, n_simulations, max_steps, n_episodes, training, rng), maxiter=n_calls, popsize=popsize, disp=True)  
#esult_timefinal = (time.time() - result_time)/60
#rint("The optimization takes", result_timefinal, "mins")

print("result.x, result.fun", result.x, result.fun)

## FINAL PARAMETER RETRIEVAL ##
# Retrieve the optimized parameters in the transformed space
unbounded_temp, unbounded_gamma, unbounded_kappa = result.x

# Apply inverse transformations to obtain the original bounded parameters
inverse_temp = np.exp(unbounded_temp)             # For [0, +inf) bounded
gamma = 1 / (1 + np.exp(-unbounded_gamma))        # For [0, 1] bounded
kappa_vs = np.exp(unbounded_kappa) 

print(f"Optimized parameters: beta = {inverse_temp}, alpha = {alpha}, gamma = {gamma}, kappa_vs = {kappa_vs}")
opti_params = {"beta": inverse_temp,
               "alpha": alpha,
               "gamma": gamma,
               "kappa": kappa_vs}

# If folder does not exist, create it
if not os.path.exists('saved/opti_results'):
    os.makedirs('saved/opti_results')
with open('saved/opti_results/mfree_vshaping.json', 'w') as json_file:
    json.dump({'opti_params': opti_params, 'fun': result.fun}, json_file)