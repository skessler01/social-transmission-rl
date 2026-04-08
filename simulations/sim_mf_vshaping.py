import os
import numpy as np
import json

from utils.plot_functions import plot_performance
from utils.social_functions import social_sim_mf
from models.mf_valueshaping import mf_valueshaping

"""Simulation of the model-free value shaping agent."""

# Simulation parameters
save = True
n_episodes = 20
world_model = "exp3" # "baseline" "exp3" "exp2"
training = 0.5
n_states = 100
n_simulations = 1000 #1000
max_steps = 40
n_actions = 4

seed = 5
rng = np.random.default_rng(seed)

## LOAD DATA ##
# Load parameters
print("Params agent value shaping:")
with open(f'saved/opti_results/mfree_vshaping.json', 'r') as json_file:
    params = json.load(json_file)["opti_params"]
print(params)

# Load data from the expert
with open('saved/baseline/mbased_expert_baseline.json', 'r') as json_file:
    expert_data= json.load(json_file)
# Convert the saved lists to array
for k in expert_data.keys():
    expert_data[k] = np.array(expert_data[k])

# Load rewards and worlds
loaded = np.load('saved/worlds.npz')
worlds_saved = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

rewards_load = np.load('saved/rewards_info.npz')
rewards_shuffled = [rewards_load[f'arr_{i}'] for i in range(len(rewards_load.files))]

if world_model == "exp2":
    rewards_load_exp2 = np.load('saved/rewards_exp2.npz')
    rewards_exp2 = [rewards_load_exp2[f'arr_{i}'] for i in range(len(rewards_load_exp2.files))]
else:
    rewards_exp2 = None

# Load agent function
agent_function = mf_valueshaping

## SIMULATION ##
print(f"MF Value Shaping simulations with {world_model} world model")
final_value_saved, sum_rewards, states_saved, actions_saved, steps_to_reward, value_epi_saved, rewards_result_steps = social_sim_mf(agent_function, expert_data, worlds_saved, rewards_shuffled, n_simulations, max_steps, n_episodes, params, training, rng, optimization = False, world_model=world_model,rewards_exp2=rewards_exp2)


## SAVE DATA ##
data = {"sum_rewards": sum_rewards.tolist(), 
                "steps_to_reward": steps_to_reward.tolist(), 
                "value": final_value_saved.tolist(), 
                "states_saved": states_saved.tolist(),
                "actions_saved": actions_saved.tolist()
                }

if save:
    # Create directory if it doesn't exist
    saving_path = f'saved/{world_model}'
    os.makedirs(saving_path, exist_ok=True)

    with open(f'saved/{world_model}/mfree_vshaping_{world_model}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    np.savez_compressed(f'saved/{world_model}/vshaping_mfree_value_epi.npz', *value_epi_saved)
    print("MF Social VS data saved")

    ## PLOT PERFORMANCE ##
    fig1, ax1 = plot_performance(sum_rewards, "Episodes", f"MF Value Shaping {params['kappa']} kappa", "Performance", n_episodes*training)
    fig1.savefig(f'saved/figures/{world_model}/mf_vshaping_{world_model}_performance.png')
    fig2, ax2 = plot_performance(steps_to_reward, "Episodes", f"MF Value Shaping {params['kappa']} kappa", "Steps to reward", n_episodes*training )    
    fig2.savefig(f'saved/figures/{world_model}/mf_vshaping_{world_model}_steps_to_reward.png')