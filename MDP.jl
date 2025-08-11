
struct MDP
    horizon::Int
    actions::Vector{Int}
    states::Vector{Int}
    reward_function::Function # a function of (s,a)
    gamma::Real
    dynamics::Array{Float64, 3} # [s',s,a] entry corresponds to p(s'|s,a)
end 

function random_MDP(num_states::Int, num_actions::Int; γ = 0.95, deterministic = false, horizon::Int = Inf)::MDP
    """
    Generates a random MDP 
    """
    states = collect(1:num_states)
    actions = collect(1:num_actions)

    # Random dynamics: shape (s′, s, a)
    dynamics = zeros(Float64, num_states, num_states, num_actions)
    if deterministic
        for s in 1:num_states, a in 1:num_actions
            next_state = rand(1:num_states)
            for s′ in 1:num_states 
                dynamics[s′, s, a] = (s′ == next_state)
            end
        end
    else
        for s in 1:num_states, a in 1:num_actions
            probs = rand(num_states)
            dynamics[:, s, a] .= probs ./ sum(probs)  # Normalize to sum to 1
        end
    end

    # Random reward function: r(s, a)
    rewards = randn(num_states, num_actions)
    reward_fn = (s, a) -> rewards[s, a]

    return MDP(horizon, actions, states, reward_fn, γ, dynamics)
end

function random_rollout(mdp::MDP)
    """
    Generates a random rollout policy given an MDP 
    """
    num_states = length(mdp.states)
    num_actions = length(mdp.actions)

    rollout_policy = zeros(Float64, num_states, num_actions)
    for s in 1:num_states
        probs = rand(num_actions)
        rollout_policy[s, :] .= probs ./ sum(probs)
    end
    return rollout_policy  # shape (states × actions)
end

function ucb(v::Real, n::Int, t::Int, c_param::Real=1.0)
    if n == 0
        return Inf  # force exploration of unvisited actions
    elseif t == 0
        return -Inf # no parent visits means no basis for calculation; or handle differently
    else
        average_reward = v / n
        exploration = c_param * sqrt(2 * log(t) / n)
        return average_reward + exploration
    end
end

# function r(s::Int, a::Int)
#     """
#     Basic reward function
#     """
#     return s == 2 ? 1.0 : 0.0
# end

# # State and action space
# States = [1, 2, 3]
# Actions = [1, 2]

# # [a,s,s_next]
# raw = [
#     # Action = 1 
#     0.8, 0.2, 0.0, # s = 1, varying s'
#     0.1, 0.8, 0.1, # s = 2, varying s'
#     0.0, 0.2, 0.8, # s = 3, varying s'

#     # Action = 2
#     0.2, 0.8, 0.0, # s = 1, varying s'
#     0.0, 0.2, 0.8, # s = 2, varying s'
#     0.0, 0.8, 0.2  # s = 3, varying s'
# ]

# # now (next_states, state, actions)
# dynamics = reshape(raw, 3, 3, 2)      


# rollout_policy = reshape([
#     0 1 
#     1 0
#     0 1
# ], (3,2))





