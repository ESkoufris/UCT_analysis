##########################
# MCTS running functions #
##########################
include("project.jl")
include("MCTS.jl")
include("value_iteration.jl")


using Plots

function run_MCTS(; 
    mdp=nothing,                  # supply your own MDP or leave as nothing
    rollout_type::Symbol=:random, # :random, :optimal, or :custom
    rollout_policy_custom=nothing,
    gamma=1,
    max_depth=50,
    c_param=1.2,
    S=6,
    A=3,
    H=10,
    tries=100,
    mcts_iterations=10_000
)
    # Create or use given MDP
    if isnothing(mdp)
        randMDP = random_MDP(S, A, γ=gamma, is_deterministic=true, horizon=H)
    else
        randMDP = mdp
        S, A = length(randMDP.states), length(randMDP.actions)
    end

    # Get optimal policy if needed
    optimal_policy = value_iteration(randMDP)[2]

    # Choose rollout policy
    if rollout_type == :random
        rollout_policy = random_rollout(randMDP)
    elseif rollout_type == :optimal
        rollout_policy = zeros(S, A)
        for s in 1:S, a in 1:A
            rollout_policy[s, a] = (optimal_policy[s] == a)
        end
    elseif rollout_type == :custom
        if isnothing(rollout_policy_custom)
            error("Custom rollout policy not provided")
        end
        rollout_policy = rollout_policy_custom
    else
        error("Invalid rollout_type: choose :random, :optimal, or :custom")
    end

    # Config for MCTS
    config = MCTSConfig(randMDP, rollout_policy, H, ucb, c_param)

    # Store proportion of correct actions per trial
    proportions = zeros(tries)

    for t in 1:tries
        correct_count = 0
        for initial_state in 1:length(randMDP.states)
            root = MCTSNode(initial_state)
            best_act = best_actions(root, config, mcts_iterations)
            if best_act == optimal_policy[initial_state]
                correct_count += 1
            end
        end
        proportions[t] = correct_count / length(randMDP.states)
    end

    # Plot proportion of correct actions over time
    p = plot(1:tries, proportions,
         xlabel="Trial", ylabel="Proportion Correct",
         title="MCTS Accuracy Over Time", legend=false, ylim=(0,1))
    display(p)
    
    return proportions, randMDP, optimal_policy
end



################################
# Drift and distribution shift #
################################