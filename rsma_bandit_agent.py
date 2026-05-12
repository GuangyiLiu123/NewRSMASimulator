import numpy as np

from rsma_minimal_sim import SimulationConfig, RSMASimulator, RSMAAction


def make_action_set(K: int):
    actions = []

    for pc in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        private_total = 1.0 - pc

        actions.append(
            RSMAAction(
                power_common=pc,
                power_private=np.full(K, private_total / K),
                common_split=np.full(K, 1.0 / K),
            )
        )

    return actions


def evaluate_round_robin(num_trials_per_action=10000):
    K = 5

    cfg = SimulationConfig(
        n_tx=4,
        n_users=K,
        snr_db=10.0,
        num_trials=1,
        rng_seed=42,
        csit_error_var=0.00,
        pathloss=np.linspace(1.0, 8.0, K),
        lambda_sum_rate=0.7,
    )

    groups = [
        [0, 1],
        [2, 3, 4],
    ]

    sim = RSMASimulator(cfg, groups=groups)
    action_set = make_action_set(K)

    rewards = np.zeros(len(action_set))
    counts = np.zeros(len(action_set))

    for action_index, action in enumerate(action_set):
        for _ in range(num_trials_per_action):
            state = sim.sample_state()

            result = sim.evaluate_action(
                h=state["h"],
                h_hat=state["h_hat"],
                action=action,
            )

            # Choose what the bandit is optimizing:
            rewards[action_index] += result.grouped_reward
            # alternatives:
            # rewards[action_index] += result.grouped_sum_rate
            # rewards[action_index] += result.sum_rate
            # rewards[action_index] += result.sdma_reward

            counts[action_index] += 1

    avg_rewards = rewards / counts

    print("\nRound-robin action evaluation:")
    for i, action in enumerate(action_set):
        print(
            f"Action {i}: "
            f"Pc={action.power_common:.2f}, "
            f"Ppriv={np.round(action.power_private, 3)}, "
            f"Csplit={np.round(action.common_split, 3)} "
            f"-> avg reward={avg_rewards[i]:.4f}, "
            f"tested {int(counts[i])} times"
        )

    best_index = int(np.argmax(avg_rewards))
    best_action = action_set[best_index]

    print("\nBest round-robin action:")
    print(best_action)


if __name__ == "__main__":
    evaluate_round_robin(num_trials_per_action=10000)