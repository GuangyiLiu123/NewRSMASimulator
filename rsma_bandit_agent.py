import numpy as np

from rsma_minimal_sim import SimulationConfig, RSMASimulator, RSMAAction


ACTION_SET = [
    RSMAAction(0.1, 0.45, 0.45, 0.5),
    RSMAAction(0.2, 0.40, 0.40, 0.5),
    RSMAAction(0.3, 0.35, 0.35, 0.5),
    RSMAAction(0.4, 0.30, 0.30, 0.5),
    RSMAAction(0.5, 0.25, 0.25, 0.5),
    RSMAAction(0.8, 0.10, 0.10, 0.5),
]


def evaluate_round_robin(num_trials_per_action=10000):
    cfg = SimulationConfig(
        n_tx=4,
        snr_db=10.0,
        num_trials=1,
        rng_seed=42,
        csit_error_var=0.00,
        pathloss_user_1=1.0,
        pathloss_user_2=2.0,
    )

    sim = RSMASimulator(cfg)

    rewards = np.zeros(len(ACTION_SET))
    counts = np.zeros(len(ACTION_SET))

    for action_index, action in enumerate(ACTION_SET):
        for _ in range(num_trials_per_action):
            state = sim.sample_state()

            result = sim.evaluate_action(
                h1=state["h1"],
                h2=state["h2"],
                h1_hat=state["h1_hat"],
                h2_hat=state["h2_hat"],
                action=action,
            )

            rewards[action_index] += result.sum_rate
            counts[action_index] += 1

    avg_rewards = rewards / counts

    print("\nRound-robin action evaluation:")
    for i, action in enumerate(ACTION_SET):
        print(
            f"Action {i}: "
            f"Pc={action.power_common:.2f}, "
            f"P1={action.power_private_1:.2f}, "
            f"P2={action.power_private_2:.2f}, "
            f"alpha={action.common_split_alpha:.2f} "
            f"-> avg reward={avg_rewards[i]:.4f}, "
            f"tested {int(counts[i])} times"
        )

    best_index = int(np.argmax(avg_rewards))
    best_action = ACTION_SET[best_index]

    print("\nBest round-robin action:")
    print(best_action)


if __name__ == "__main__":
    evaluate_round_robin(num_trials_per_action=10000)