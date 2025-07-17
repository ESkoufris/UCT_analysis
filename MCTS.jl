include("project.jl")
include("MDP.jl")

struct MCTSConfig
    MDP::MDP
    rollout_policy::Array
    max_depth::Int
    score_function::Function
    c_param::Real
end

mutable struct MCTSNode
    state
    parent::Union{MCTSNode, Nothing}
    children::Dict{Int, MCTSNode}
    visits::Int
    value::Float64
    depth::Int
end

function MCTSNode(state, parent::Union{MCTSNode, Nothing}=nothing)
    children = Dict()
    visits = 0
    value = 0.0
    depth = parent === nothing ? 0 : parent.depth + 1
    return MCTSNode(state, parent, children, visits, value, depth)
end

function is_fully_expanded(Node::MCTSNode, config::MCTSConfig)
    if length(Node.children) == length(config.MDP.actions)
        return true
    else 
        return false
    end
end

function best_child(node::MCTSNode, config::MCTSConfig)
    """Used during in-tree pahse to select child-node"""
    @assert is_fully_expanded(node, config)
    score_function = config.score_function
    children = collect(values(node.children))
    scores = [
        score_function(child.value, child.visits, node.visits, config.c_param) for child in children 
    ]
    action = argmax(scores)
    return action, node.children[action]
end 

function Base.show(io::IO, node::MCTSNode)
    print(io, "d=$(node.depth) n=$(node.visits) v=$(node.value)")
end

function sample_next_state(dynamics, state, action)
    """
    Used in expansion phase to select random unvisited action 
    """
    probs = dynamics[:, state, action]
    dist = Categorical(probs)
    next_state = rand(dist)
    return next_state
end

function sample_next_action(rollout_policy, state)
    """
    Used in rollout phase to simulate return following a rollout policy
    """
    probs = rollout_policy[state, :]
    dist = Categorical(probs)
    action = rand(dist)
    return action
end

function simulate(state, config::MCTSConfig, depth)
    """Simulate a rollout from leaf node using rollout policy"""
    total_reward = 0.0
    for d in 1:depth
        action = sample_next_action(config.rollout_policy, state)
        
        # get reward 
        reward = config.MDP.reward_function(state, action)
        total_reward += config.MDP.gamma^d * reward

        # sample next state 
        next_state = sample_next_state(config.MDP.dynamics, state, action)

        state = next_state
        if d == config.max_depth
            break 
        end 
    end    
    return total_reward
end

function expand(node::MCTSNode, config::MCTSConfig)
    """Expand the node by trying an untried action and creating a new child."""
    tried_actions = collect(keys(node.children))
    untried_actions = [a for a in config.MDP.actions if a ∉ tried_actions]
    @assert !isempty(untried_actions)

    # sample random action from untried actions 
    action = rand(untried_actions)

    # get immediate reward 
    reward = config.MDP.reward_function(node.state, action)

    # Sample the next state based on transition probabilities
    next_state = sample_next_state(config.MDP.dynamics, node.state, action)

    # create child node
    child_node = MCTSNode(next_state, node)
    node.children[action] = child_node 
    return child_node, reward
end

function backpropagate(trajectory::Array, rollout_reward::Real, config::MCTSConfig)
    """
    Backpropagate rewards from a rollout through the trajectory 
    """
    cumulative_reward = rollout_reward
    for (node, reward) in reverse(trajectory)
        cumulative_reward = reward + config.MDP.gamma * cumulative_reward
        node.visits += 1
        node.value += cumulative_reward
    end
end

function best_actions(root::MCTSNode, config::MCTSConfig, iterations=100)
    """
    Estimate and return the best action from a root node 
    """
    for _ in 1:iterations
        node = root
        trajectory = []

        while is_fully_expanded(node, config) && !isempty(node.children) && node.depth < config.max_depth
            (action, child) = best_child(node, config)
            ## FIX this 
            reward = config.MDP.reward_function(node.state, action)
            push!(trajectory, (node, reward))
            node = child
        end

        if !is_fully_expanded(node, config) && node.depth < config.max_depth
            child, expanded_reward = expand(node, config)
            push!(trajectory, (node, expanded_reward))
            node = child
        end

        rollout_reward = simulate(node.state, config, config.max_depth - node.depth)
        backpropagate(trajectory, rollout_reward, config)
    end
    # if root.state == 2 
    #     println("Action 1", root.children[1])
    #     println("Action 2", root.children[2])
    # end

    values = [root.children[action].value / root.children[action].visits for action in config.MDP.actions]
    best_action = argmax(values)
    return best_action
end
