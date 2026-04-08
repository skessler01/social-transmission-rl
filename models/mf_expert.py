import numpy as np
from utils.helper_functions import softmax_policy, q_learning, find_reward


"""Policy of model-free ascoial agent/expert."""

# Model Free Expert Q-Learning
def expert_ql(params, env, world, rewards_info, max_steps, n_episodes, training, rng, optimization, agent = "agent", world_model='baseline', rewards_exp2= None): # params

    # Assert that exp is either 'baseline' or 'exp2' or 'exp3'
    assert world_model in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"

    # Start with a uniform value function
    value = np.ones((env.n_states, env.n_actions)) # value matrix
    # To save the value function for each episode
    value_perepi = np.zeros((n_episodes, env.n_states, env.n_actions))


    # Initialize the reward mat
    reward_sums_steps = np.zeros((n_episodes, max_steps))
    reward_sums_epi = np.zeros((n_episodes))
    
    # Initialize state mat as 
    state_mat = np.nan*np.zeros((n_episodes, max_steps+1), dtype=int)
    
    # Initialize action mat as 
    action_mat = np.nan*np.zeros((n_episodes, max_steps))
    
    
    steps_to_reward = np.zeros((n_episodes,))
    
    idx = 0
    # Loop over episodes
    for episode in range(n_episodes):

        reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]

        if episode >= (n_episodes * training) and agent == "agent":
            if world_model == 'exp2':
                if rewards_exp2 is not None:
                    reward_placed = rewards_exp2[-n_episodes:, :][episode][:,:2]
                else:
                    raise ValueError("rewards_exp2 must be provided for exp2")

            # Possible initial loc        
            agent_location = env.initial_loc(exp = world_model)
        else:
            agent_location = env.initial_loc(exp = "baseline")

        state = world[agent_location]
        # save initial state
        state_mat[episode, 0] = state
        
        # Loop over steps
        for t in range(max_steps):

            # Chose the next action randomly
            _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)
            
            # Calculate new location based on the action
            # Check if transition is possible and move
            next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)
            
            # observe reward for that action 
            reward = find_reward(state, reward_placed)  
            # Update rewards based in agent location
            reward_sums_steps[episode, t] = reward
            reward_sums_epi[episode] += reward
            
            ## -- APPLY THE LEARNING RULE - Q LEARNING --
            
            value = q_learning(value, state, action, next_state, reward, params['gamma'], params['alpha'])
            ## -- End of Learning Rule --

        
            ## -- UPDATES --
            
            action_mat[episode,t] = action
            
            # FOR THE TERMINATION EPISODE
            if reward > 0:
                steps_to_reward[episode] = t + 1
                break  # episode ends
            
            # Update agent location and state
            state = next_state
            agent_location = next_agent_location

            # Save the state
            state_mat[episode, t+1] = state
        # If the episode doesnt terminate, it locates a 40
        steps_to_reward[episode] = t + 1

    if optimization:
        return reward_sums_epi
    
    else:

       return value, reward_sums_epi, state_mat, action_mat, steps_to_reward, value_perepi, reward_sums_steps