# Emergent social transmission of model-based representations without inference
Code for the simulations and analyses from “Emergent social transmission of model-based representations without inference” (arXiv:2604.05777), demonstrating how simple social cues can drive the emergence of model-based representations. 

## How to run the simulations
To run the simulation of an agent, run *python -m simulations.sim_typeoflearning_typeofagent* from the root folder. For example, simulate the model-based decision-bias agent, run the file *sim_mb_dbias.py*.
At the top of each file, you can specify the following parameters: 
* *save*: whether to save the results of the simulation
* *n_episodes*: number of episodes that are peformed by the agent (default = 20 for agents, 120 for expert)
* *world_wodel*: the type of world the agent acts in, i.e. the type of experiment that is run ("baseline"/"exp2"/"exp3")
* *training*: the ratio of the total episodes used for training (default = 0.5)
* *n_states*: number of states in the environment (100 in this set up)
* *n_simulations*: number of simulations (default = 1000)
* *max_steps*: the maximum number of steps or actions the agent can take before the episode ends (in case they have not found the reward)
* *seed*: seed to initialize the random number generator for reproducibility (set to 5)

For the asocial agents *sim_mb_expert.py* and *sim_mf_expert.py* set the number of episodes to 20 and *expert=False* to run the agent (i.e. the learner) and *n_episodes=120* and *expert=True* to run the expert.

The simulation files load the selected worlds and rewards, loop over the simulations and store the results from the simulations (sum_rewards, steps_to_reward, value, states_saved, actions_saved, model_saved).

In each simulation, a new world is configured and reward values are assigned to the reward states. At the start of each episode, the agent is placedrandomly in one of four start states and random noise is applied to the rewards. The agent moves between adjacent states by executing actions (up, down, left, right) and is constrained by walls: attempts to move into a wall result in the agent remaining in the current state. Actions that do not yield a positive reward incurs a cost (negative reward) of −1. An episode terminates either when a positive reward is obtained or after the maximal number of actions is performed. During the *training* phase, social learning agents incorporate expert information into their decisions.

If *save=True*, the results are stored as a .json file in the *saved* folder. Additionally, the transition matrix and value representation of the agent is saved as .npz files and performance and and the steps used to aquire a reward are plotted and saved in the *saved/figures* folder.

## Experiments
The type of experiment can be selected via *world_wodel="baseline"/"exp2"/"exp3"*:

* **Exp. 1: Baseline**: Set-up of environment and rewards as described, used to establish baseline performance
* **Exp. 2: Value swap**: In the *test* phase, two randomly selected reward states are swapped
* **Exp. 3: New start location**: In the *test* phase, the start states of the agents are shifted by one tile towards the corners (excluding reward states)

For Exp. 2, rewards are loaded from *rewards_exp2.npz* during the *test* phase.

## Repository Structure


### models
The name of the models follows the same rule as for the simulations *typeoflearning_typeofagent.py*. There *typeoflearning* refers to model based (mb) or model free (mf) reinforcement learning (RL). MB agents maintain an internal belief about the environment and use DYNA-Q to simulate experiences. There are three types of agents: asocial, decision biasing (dbias) and value shaping (vshaping). In *decision biasing*, the learner is bias towards choosing the action that minimizes the distance to the expert, in *value shaping* the learner adds a value bonus to actions performed by the expert. Note that asocial agents do not have an explicit *typeofagent* decriptive component. 

MF expert and agent use the same model, BUT, when running the simulations, it is important to change the number of episodes and set the *expert* parameter correctly (see How to run).

The general structure of all the models is:

* Initialize saving structures
* Loop over episodes
  * Extract the rewards for that simulation and episode
  * Initialize the agent with initial location
  * Find the initial state
  * Save initial state
  * Loop over steps
    * Softmax action
    * Move the agent
    * Find the reward for that state
    * Q-learning
    * If MB - dyna-q planning
    * Save state and action
    * If the agent finds the reward - termination step

For social models, the social learning is implemented inside the model. 

The models return:

* *value*: final value for the simulation. np.array (100, 4)
* *reward_sums_epi*: the total sum of reward for each episode - np.array(n_episodes)
* *state_mat*: saved all the states the agent experiences. It is useful to plot the path it has follows. np.array(n_episodes, max_steps+1). The +1 is to save the initial state as well. It is initialized as nan, so if the agent find the reward in some step, the rest of the array is np.nan
* *action_mat*: saved all the actions taken by the agent. It is useful to check how the agents follow the expert. np.array(n_episodes,max_steps). It is also initialized as np.nan
* *steps_to_reward*: number of steps taken by the agent until the reward has been found. np.array(n_episodes, )
* *value_perepi*: value saved for each episode. This is very big and was not used at the end. np.array(n_episodes, 100, 4)
* *reward_sums_steps*: serves the same purpose as reward_sums_epi, but this inidicates the reward found in each step. np.array(n_episodes, max_steps)

### optimizations
The model's hyperparameters are optimized using the differential evolution algorithm to maximize performance across the simulations in the baseline environment. 

Parameter space: 

* *beta* (0, 100): inverse temperature parameter for softmax
* *alpha* (0, 1): learning rate for q learning update
* *gamma* (0, 1): discount factor for q learning update
* *lambda* (1, 40): number of planning steps for model based learning
* *alpha_t* (0, 1): learning rate for the transition matrix
* *omega* (0, 1): social parameter for decision biasing
* *kappa* (0, 100): social parameter for value shaping

Candidate parameters are optimized in a transformed, unbounded space to enforce valid ranges. Note that Learning rates alpha and alpha_t for social learners are fixed in the current set-up to values optimized for asocial learners, due to joint optimization consistently converging to low learning rates that suppressed individual learning and led to
poor test performance.

To run an optimization, run *opti_typeoflearning_typeofagent_runsbatch.py*. Optimizations are best run via a cluster. The model based expert is the one that takes the most time (~1 and a half day). The rest should be done between 3 to 6 hours. It returns a log script where the optimized parameters and the mean performance across all episodes and simulations are printed.

### saved
This folder contains the results of the optimizations and simulations, generated figures as well as worlds, transition matrices and rewards for each simulation and episodes that are configurated before hand to ensure that all agents move in the save environment with the same rewards (see world_plots.ipynb for the generation of worlds and rewards). 

The results of the simulations can be found in the respective subfolder of the experiment. The folders contain the output of the models (see How to run the simulations), the value and transition matrices as .npz files. The transition matrices are stored in the extra folder *tmss*.

The values of the optimized hyperparameters are stored in the subfolder *opti_results*.

Rewards: 

* *rewards_fixed.npz*: contains the rewards assigned to the designated rewards states (simulations, episodes, states, reward)
* *rewards_info.npz*: contains the same rewards with added noise
* *rewards_exp2.npz*: contains the same rewards, but for each episode, two randomly selected rewards states swap their values

Transition matrices: 

* *true_tms.npz*: This is the true transition matrix for each world of the shape n_simulationsx(n_states,n_actions,n_states), **considering** walls
* *init_tms.npz*: This is the initial transition matrix for each world **without** walls.

Worlds: 

+ *worlds.npz*: The world configuration for each simulation.

Note that (non-expert) agents only have access to the last 20 episodes.

### simluations
Contains the scripts to run the simulations of the different models (see How to run).

### utils 

#### **helper_functions.py** : 
This file contains all the necessary functions to run the models:

* *softmax_policy*: returns a probability distribution over actions and an actions selected based on that distribution
* *find_reward*: find the reward value of the current state
* *assign_fixed_rewards*: randomly assign fixed reward values for each reward state
* *sample_reward_offset*: samples an integer reward offset with mean 0 and variance = target_var, using a shifted Binomial(n, 0.5)
* *q_learning*: performs the q_learning update rule
* *social_policy*: For the decision biased agent. Finds the action that reduces the distanc (number of steps) to the expert. 
* *decision_bias*: returns the action taken by the agent. It is stochastic, can take the action that reduces the distance given by *social_policy* or a softmax_action. It is given by the 'lambda' parameter. 
* *value_shaping*: Adds a bonus to the value of the agent for the expert's observed state and action
* *model_update*: Updates the model for model-based learning
* *dynaq_planner*: Dyna q update. Goes over the already visited state and action pairs and does q_learning update
* *load_data*:  Loads data from a json file

**NOTE**: Some of the functions are optimized with Numba (@njit). Numba is very picky and can't deal with dictionaries. Thats why in many of the functions the params dictionary can't be introduced, but the actual parameter. 

#### **plot_functions.py** : 
This file contains the functions to plot. The most important ones are:

* *plot_world*: visualizes the world of a selected episode
* *plot_figure1*: can be used to recreate Figure 1
* *all_models*: plots either performance (sum of rewards) or number of steps taken for the 6 models
  
Other plots for analyses can be found in the *results_plots.ipynb* notebook.

#### **social_functions.py**:
This file contains functions to run the simulation and saving the results. These functions are called in the simulation scripts. There are different functions for model_free and model_based.

#### **tile.py**:
Contains the class of Tile that defines the four 5×5 quadrants of the world:

* *TileA*: states from 0 to 24
* *TileB*: states from 25 to 49
* *TileC*: states from 50 to 74
* *TileD*: states from 75 to 99

#### transfer_metrics.py
This file contains functions used to perform the value and belief transfer metrics:

* *value_correlation* : computes the correlation of the values between two agents (normally expert and any agent). It takes distances to reward, so first calculates the distance of each state to the nearest rewards using the true transition matrix (taking into account the boundaries). For all the states with the same distances, it flattens the values and computes the spearman correlation. Returns the correlation np.array(n_simulations, max_distance)
* *z_transform_intervals*: used to compute the mean of the correlations and the SEM. Transforms the correlations in z and then back.
* *compute_true_value_function*: generates true value function of the environment using the Bellman equation
* *tm_ztransform_distance*: computes the correlation between transition matrices for distances to reward
* *normalize_tm_correlation*: normalizes agent correlations relative to baseline
* *tm_agent_similarity_over_distances*: compute learner–expert TM similarity per simulation

#### **world.py**
Contains the class VillageWorld that defines the environment. The environment is composed of four tiles (see *tile.py*). Boundaries are given as a list of tuples of the states between which the boundaries are. Reward states are fixed: [16, 43, 57, 87].

When initializing the class, if no world matrix is introduced, a random one will be generated. If called with a world_matrix, it will return everything belonging to that matrix. 

Functions:

* *get_world_matrix*: generates a random matrix rotating and placing the tiles. 
* *move_agent*: gives the next state given the current state and action of the agent, checking if it is a boundary and to which tile the state belongs to. 
* *calulcate_new_location*: returns the next location of the agent
* *get_all_boundaries*: returns all the boundaries of the tiles, that are given as attribute in the world. It is useful for plotting. 
* *transition_probabilities*: returns the initial transition matrix and the true (taking into account boundaries) transition matrix. 
* *initial_loc*: Randomly selects a initial location for the agent

### Notebooks

#### **world_plots.ipynb**
This notebook is used to generate and visualize worlds and to to generate and modify rewards.

#### **results_plots.ipynb**
Plots results of the simulations and analyses including performance, value and belief transfer.
