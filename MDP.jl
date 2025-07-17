
struct MDP
    actions::Array
    states::Array
    reward_function::Function # a function of (s,a)
    gamma::Real
    dynamics::Array # [s',s,a] entry corresponds to p(s'|s,a)
end 

function random_MDP(num_states::Int, num_actions::Int; γ = 0.95)
    """
    Generates a random MDP 
    """
    states = collect(1:num_states)
    actions = collect(1:num_actions)

    # Random dynamics: shape (s′, s, a)
    dynamics = zeros(Float64, num_states, num_states, num_actions)
    for s in 1:num_states, a in 1:num_actions
        probs = rand(num_states)
        dynamics[:, s, a] .= probs ./ sum(probs)  # Normalize to sum to 1
    end

    # Random reward function: r(s, a)
    rewards = randn(num_states, num_actions)
    reward_fn = (s, a) -> rewards[s, a]

    return MDP(actions, states, reward_fn, γ, dynamics)
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


function ucb(v::Real, n::Int, t::Int, c_param::Real=1)
    """
    t visits to state node, n visits to an action under the state node, total value v
    """
    return t == 0 || t == 1 ? Inf : (v/n + c_param * sqrt(2 * log(t) / n))
end

function r(s::Int, a::Int)
    """
    Basic reward function
    """
    return s == 2 ? 1.0 : 0.0
end

# State and action space
States = [1, 2, 3]
Actions = [1, 2]

# [a,s,s_next]
raw = [
    # Action = 1 
    0.8, 0.2, 0.0, # s = 1, varying s'
    0.1, 0.8, 0.1, # s = 2, varying s'
    0.0, 0.2, 0.8, # s = 3, varying s'

    # Action = 2
    0.2, 0.8, 0.0, # s = 1, varying s'
    0.0, 0.2, 0.8, # s = 2, varying s'
    0.0, 0.8, 0.2  # s = 3, varying s'
]

# now (next_states, state, actions)
dynamics = reshape(raw, 3, 3, 2)      

# rollout policy [s,a]
# rollout_policy = reshape([
#     0.6 0.4 
#     0.9 0.1
#     0.5 0.5
# ], (3,2))

rollout_policy = reshape([
    0 1 
    1 0
    0 1
], (3,2))

gamma = 0.9  # Discount factor for future rewards
max_depth = 30  # Maximum tree depth
c_param = 0.4 



