include("MDP.jl")

# function value_iteration(MDP; θ = 1e-6, max_iters = 1000)
#     num_states = length(MDP.states)
#     num_actions = length(MDP.actions)
#     V = zeros(num_states)
#     π = zeros(Int, num_states)
#     γ = MDP.gamma
#     H = MDP.horizon

#     for iter in 1:max_iters
#         Δ = 0.0
#         V_new = similar(V)

#         for s in 1:num_states
#             q_values = zeros(num_actions)
#             for a in 1:num_actions
#                 for s′ in 1:num_states
#                     p = MDP.dynamics[s′, s, a]
#                     r = MDP.reward_function(s, a)
#                     q_values[a] += p * (r + γ * V[s′])
#                 end
#             end
#             best_value, best_action = findmax(q_values)
#             V_new[s] = best_value
#             π[s] = best_action
#             Δ = max(Δ, abs(V_new[s] - V[s]))
#         end

#         V = V_new
#         if Δ < θ
#             break
#         end
#     end
#     return V, π
# end

function value_iteration(MDP; θ = 1e-6, max_iters = 1000)
    num_states = length(MDP.states)
    num_actions = length(MDP.actions)
    γ = MDP.gamma
    H = MDP.horizon

    if isinf(H)
        # ----- Infinite horizon: standard fixed point iteration -----
        V = zeros(num_states)
        π = zeros(Int, num_states)

        for iter in 1:max_iters
            Δ = 0.0
            V_new = similar(V)

            for s in 1:num_states
                q_values = zeros(num_actions)
                for a in 1:num_actions
                    for s′ in 1:num_states
                        p = MDP.dynamics[s′, s, a]
                        r = MDP.reward_function(s, a)
                        q_values[a] += p * (r + γ * V[s′])
                    end
                end
                V_new[s], π[s] = findmax(q_values)
                Δ = max(Δ, abs(V_new[s] - V[s]))
            end

            V = V_new
            if Δ < θ
                break
            end
        end
        return V, π

    else
        # ----- Finite horizon: backward induction -----
        V = zeros(H+1, num_states)  # V[t, s] = value with t steps remaining
        π = zeros(Int, num_states)

        for t in 1:H
            for s in 1:num_states
                q_values = zeros(num_actions)
                for a in 1:num_actions
                    for s′ in 1:num_states
                        p = MDP.dynamics[s′, s, a]
                        r = MDP.reward_function(s, a)
                        q_values[a] += p * (r + γ * V[t, s′])  # look back one step
                    end
                end
                V[t+1, s], π[s] = findmax(q_values)
            end
        end
        return V[H+1, :], π
    end
end