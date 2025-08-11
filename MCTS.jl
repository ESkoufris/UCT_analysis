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
    action
    parent::Union{MCTSNode, Nothing}
    children::Dict{Int, MCTSNode}
    visits::Int
    value::Float64
    depth::Int
end

function MCTSNode(state = nothing, action = nothing, parent::Union{MCTSNode, Nothing}=nothing)::MCTSNode
    children = Dict{Int,MCTSNode}()
    visits = 0
    value = 0.0
    depth = parent === nothing ? 0 : parent.depth + 1
    return MCTSNode(state, action, parent, children, visits, value, depth)
end

function is_fully_expanded(Node::MCTSNode, config::MCTSConfig)::Bool
    """
    Checks if a node has had all its children visited at least once 
    """
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
    actions = collect(keys(node.children))
    children = collect(values(node.children))
    scores = [score_function(c.value, c.visits, node.visits, config.c_param) for c in children]
    i = argmax(scores)
    # println(scores)
    return actions[i], children[i]
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
    discount = 1.0
    for d in 1:depth
        action = sample_next_action(config.rollout_policy, state)
        reward = config.MDP.reward_function(state, action)
        total_reward += discount * reward
        discount *= config.MDP.gamma
        state = sample_next_state(config.MDP.dynamics, state, action)
        if d == config.max_depth
            break
        end
    end
    return total_reward
end

function expand(node::MCTSNode, state, config::MCTSConfig)
    """Expand the node by trying an untried action and creating a new child."""
    tried_actions = collect(keys(node.children))
    untried_actions = [a for a in config.MDP.actions if a ∉ tried_actions]
    @assert !isempty(untried_actions)

    # sample random action from untried actions 
    action = rand(untried_actions)

    # # get immediate reward from action
    reward = config.MDP.reward_function(state, action)

    # create child node
    child_node = MCTSNode(nothing, nothing, node)
    node.children[action] = child_node 
    return action, child_node, reward
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
        state = node.state

        # Selection phase: traverse the tree selecting actions according to score function
        while is_fully_expanded(node, config) && !isempty(node.children) && node.depth < config.max_depth
            (action, child) = best_child(node, config)
            reward = config.MDP.reward_function(state, action)
            next_state = sample_next_state(config.MDP.dynamics, state, action)
            push!(trajectory, (node, reward)) # Come back: check if correct!
            node = child
            state = next_state
        end

        # Expansion: Expand the node if it's not fully expanded
        if !is_fully_expanded(node, config) && node.depth < config.max_depth
            action, child, expanded_reward = expand(node, state, config)
            push!(trajectory, (node, expanded_reward))
            next_state = sample_next_state(config.MDP.dynamics, state, action)
            node = child  
            state = next_state
        end


        # Simulation: Perform a random rollout from the child node 
        # of the newly expanded node 
        rollout_reward = simulate(state, config, config.max_depth - node.depth)
        backpropagate(trajectory, rollout_reward, config)
    end

    values = [
        haskey(root.children, a) ? root.children[a].value / root.children[a].visits : -Inf
        for a in config.MDP.actions
    ]

    # Choose the action leading to the best child
    best_action = argmax(values)
    return best_action
end
