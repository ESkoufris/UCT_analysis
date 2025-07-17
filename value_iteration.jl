include("MDP.jl")

function value_iteration(MDP; θ = 1e-6, max_iters = 1000)
    num_states = length(MDP.states)
    num_actions = length(MDP.actions)
    V = zeros(num_states)
    π = zeros(Int, num_states)  # policy: π[s] = best action at state s
    γ = MDP.gamma
    for iter in 1:max_iters
        Δ = 0.0
        V_new = copy(V)

        for s in 1:num_states
            best_action = 0
            best_value = -Inf

            for a in 1:num_actions
                q = 0.0
                for s′ in 1:num_states
                    p = MDP.dynamics[s′, s, a]
                    r = MDP.reward_function(s, a)
                    q += p * (r + γ * V[s′])
                end

                if q > best_value
                    best_value = q
                    best_action = a
                end
            end

            V_new[s] = best_value
            π[s] = best_action
            Δ = max(Δ, abs(V_new[s] - V[s]))
        end

        V = V_new
        if Δ < θ
            break
        end
    end

    return V, π
end