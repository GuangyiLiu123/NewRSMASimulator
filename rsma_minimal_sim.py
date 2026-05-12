import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class SimulationConfig:
    n_tx: int = 4
    n_users: int = 2
    snr_db: float = 10.0
    num_trials: int = 1000
    rng_seed: int = 7
    csit_error_var: float = 0.0
    pathloss: Optional[np.ndarray] = None

    # Reward objective
    lambda_sum_rate: float = 0.7


@dataclass
class RSMAAction:
    power_common: float
    power_private: np.ndarray
    common_split: np.ndarray

    def validate(self, n_users: int) -> None:
        if len(self.power_private) != n_users:
            raise ValueError("power_private must have length n_users")
        if len(self.common_split) != n_users:
            raise ValueError("common_split must have length n_users")
        if self.power_common < 0:
            raise ValueError("power_common must be non-negative")
        if np.any(self.power_private < 0):
            raise ValueError("private powers must be non-negative")
        if np.any(self.common_split < 0):
            raise ValueError("common_split entries must be non-negative")

        total_power = self.power_common + np.sum(self.power_private)
        if not np.isclose(total_power, 1.0, atol=1e-9):
            raise ValueError(f"Power fractions must sum to 1.0, got {total_power:.6f}")

        split_sum = np.sum(self.common_split)
        if not np.isclose(split_sum, 1.0, atol=1e-9):
            raise ValueError(f"Common split must sum to 1.0, got {split_sum:.6f}")


@dataclass
class TrialResult:
    rc_users: np.ndarray
    rc: float
    c_users: np.ndarray
    rp_users: np.ndarray
    r_users: np.ndarray
    sum_rate: float
    fairness: float
    reward: float

    sdma_rates: np.ndarray
    sdma_sum_rate: float
    sdma_fairness: float
    sdma_reward: float

    grouped_r_users: np.ndarray
    grouped_sum_rate: float
    grouped_fairness: float
    grouped_reward: float
    grouped_common_rates: List[float]

    sinr_common: np.ndarray
    sinr_private: np.ndarray
    channel_norms: np.ndarray
    channel_alignment: np.ndarray


class RSMASimulator:
    def __init__(self, config: SimulationConfig, groups: Optional[List[List[int]]] = None):
        self.cfg = config
        self.rng = np.random.default_rng(config.rng_seed)
        self._validate_config()

        self.K = self.cfg.n_users
        self.tx_power = self._db_to_linear(self.cfg.snr_db)
        self.noise_power = 1.0

        if self.cfg.pathloss is None:
            self.pathloss = np.ones(self.K)
        else:
            self.pathloss = np.asarray(self.cfg.pathloss, dtype=float)

        self.groups = groups if groups is not None else [list(range(self.K))]
        self._validate_groups(self.groups)

    def _validate_config(self) -> None:
        if self.cfg.n_tx < 1:
            raise ValueError("n_tx must be at least 1")
        if self.cfg.n_users < 2:
            raise ValueError("n_users must be at least 2")
        if self.cfg.num_trials < 1:
            raise ValueError("num_trials must be at least 1")
        if self.cfg.csit_error_var < 0:
            raise ValueError("csit_error_var must be non-negative")
        if not (0.0 <= self.cfg.lambda_sum_rate <= 1.0):
            raise ValueError("lambda_sum_rate must be in [0, 1]")

        if self.cfg.pathloss is not None:
            pathloss = np.asarray(self.cfg.pathloss, dtype=float)
            if len(pathloss) != self.cfg.n_users:
                raise ValueError("pathloss must have length n_users")
            if np.any(pathloss <= 0):
                raise ValueError("all pathloss values must be positive")

    def _validate_groups(self, groups: List[List[int]]) -> None:
        seen = sorted([u for group in groups for u in group])
        expected = list(range(self.K))

        if seen != expected:
            raise ValueError(
                f"groups must form a partition of users {expected}, got {seen}"
            )

    def default_action(self) -> RSMAAction:
        power_common = 0.2
        private_total = 1.0 - power_common

        return RSMAAction(
            power_common=power_common,
            power_private=np.full(self.K, private_total / self.K),
            common_split=np.full(self.K, 1.0 / self.K),
        )

    @staticmethod
    def _db_to_linear(db: float) -> float:
        return 10.0 ** (db / 10.0)

    def _complex_gaussian_vector(self, size: int) -> np.ndarray:
        return (
            self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        ) / np.sqrt(2.0)

    @staticmethod
    def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < eps:
            raise ValueError("Cannot normalize near-zero vector")
        return vec / norm

    @staticmethod
    def _rate_from_sinr(sinr: float) -> float:
        return float(np.log2(1.0 + max(sinr, 0.0)))

    @staticmethod
    def _jains_fairness(rates: np.ndarray) -> float:
        numerator = np.sum(rates) ** 2
        denominator = len(rates) * np.sum(rates ** 2) + 1e-12
        return float(numerator / denominator)

    def _reward(self, rates: np.ndarray) -> float:
        sum_rate = float(np.sum(rates))
        fairness = self._jains_fairness(rates)
        return float(
            self.cfg.lambda_sum_rate * sum_rate
            + (1.0 - self.cfg.lambda_sum_rate) * fairness
        )

    def _project(self, h: np.ndarray, p: np.ndarray) -> complex:
        return np.vdot(h, p)

    def sample_state(self) -> Dict[str, np.ndarray]:
        h = np.zeros((self.K, self.cfg.n_tx), dtype=complex)

        for k in range(self.K):
            h[k] = self._complex_gaussian_vector(self.cfg.n_tx) / np.sqrt(
                self.pathloss[k]
            )

        if self.cfg.csit_error_var == 0.0:
            h_hat = h.copy()
        else:
            sigma = np.sqrt(self.cfg.csit_error_var)
            error = np.zeros_like(h)

            for k in range(self.K):
                error[k] = sigma * self._complex_gaussian_vector(self.cfg.n_tx)

            h_hat = h + error

        return {"h": h, "h_hat": h_hat}

    def _build_private_dirs(self, h_hat: np.ndarray) -> np.ndarray:
        private_dirs = np.zeros_like(h_hat, dtype=complex)

        for k in range(self.K):
            private_dirs[k] = self._normalize(h_hat[k])

        return private_dirs

    def _channel_alignment_matrix(self, h_hat: np.ndarray) -> np.ndarray:
        align = np.zeros((self.K, self.K))
        norms = np.linalg.norm(h_hat, axis=1)

        for i in range(self.K):
            for j in range(self.K):
                denom = norms[i] * norms[j] + 1e-12
                align[i, j] = np.abs(np.vdot(h_hat[i], h_hat[j])) / denom

        return align

    def _evaluate_sdma(
        self,
        h: np.ndarray,
        private_dirs: np.ndarray,
        action: RSMAAction,
    ) -> np.ndarray:
        private_total = max(np.sum(action.power_private), 1e-12)

        sdma_precoders = np.zeros_like(private_dirs, dtype=complex)

        for k in range(self.K):
            sdma_power_k = self.tx_power * action.power_private[k] / private_total
            sdma_precoders[k] = np.sqrt(sdma_power_k) * private_dirs[k]

        sdma_gain = np.zeros((self.K, self.K))

        for user in range(self.K):
            for stream in range(self.K):
                sdma_gain[user, stream] = (
                    np.abs(self._project(h[user], sdma_precoders[stream])) ** 2
                )

        sdma_rates = np.zeros(self.K)

        for user in range(self.K):
            desired = sdma_gain[user, user]
            interference = np.sum(sdma_gain[user, :]) - desired
            sinr_sdma = desired / (interference + self.noise_power)
            sdma_rates[user] = self._rate_from_sinr(sinr_sdma)

        return sdma_rates

    def _evaluate_global_rsma(
        self,
        h: np.ndarray,
        h_hat: np.ndarray,
        private_dirs: np.ndarray,
        action: RSMAAction,
    ):
        common_dir = self._normalize(np.sum(private_dirs, axis=0))
        pc = np.sqrt(self.tx_power * action.power_common) * common_dir

        private_precoders = np.zeros_like(private_dirs, dtype=complex)
        for k in range(self.K):
            private_precoders[k] = (
                np.sqrt(self.tx_power * action.power_private[k]) * private_dirs[k]
            )

        gain_private = np.zeros((self.K, self.K))
        gain_common = np.zeros(self.K)

        for user in range(self.K):
            gain_common[user] = np.abs(self._project(h[user], pc)) ** 2

            for stream in range(self.K):
                gain_private[user, stream] = (
                    np.abs(self._project(h[user], private_precoders[stream])) ** 2
                )

        sinr_common = np.zeros(self.K)
        rc_users = np.zeros(self.K)

        for user in range(self.K):
            private_interference = np.sum(gain_private[user, :])
            sinr_common[user] = gain_common[user] / (
                private_interference + self.noise_power
            )
            rc_users[user] = self._rate_from_sinr(sinr_common[user])

        rc = float(np.min(rc_users))
        c_users = action.common_split * rc

        sinr_private = np.zeros(self.K)
        rp_users = np.zeros(self.K)

        for user in range(self.K):
            desired = gain_private[user, user]
            interference = np.sum(gain_private[user, :]) - desired

            sinr_private[user] = desired / (interference + self.noise_power)
            rp_users[user] = self._rate_from_sinr(sinr_private[user])

        r_users = c_users + rp_users

        return rc_users, rc, c_users, rp_users, r_users, sinr_common, sinr_private

    def _evaluate_grouped_rsma(
        self,
        h: np.ndarray,
        private_dirs: np.ndarray,
        action: RSMAAction,
    ):
        grouped_c_users = np.zeros(self.K)
        grouped_rp_users = np.zeros(self.K)
        grouped_common_rates = []

        private_precoders = np.zeros_like(private_dirs, dtype=complex)
        for k in range(self.K):
            private_precoders[k] = (
                np.sqrt(self.tx_power * action.power_private[k]) * private_dirs[k]
            )

        # Total common power is split equally across groups for now.
        common_power_per_group = action.power_common / len(self.groups)

        for group in self.groups:
            group = list(group)

            group_common_dir = self._normalize(np.sum(private_dirs[group], axis=0))
            group_common_precoder = (
                np.sqrt(self.tx_power * common_power_per_group) * group_common_dir
            )

            rc_group_users = np.zeros(len(group))

            for idx, user in enumerate(group):
                common_gain = np.abs(self._project(h[user], group_common_precoder)) ** 2

                # User treats all private streams as interference.
                private_interference = 0.0
                for stream in range(self.K):
                    private_interference += (
                        np.abs(self._project(h[user], private_precoders[stream])) ** 2
                    )

                rc_group_users[idx] = self._rate_from_sinr(
                    common_gain / (private_interference + self.noise_power)
                )

            rc_group = float(np.min(rc_group_users))
            grouped_common_rates.append(rc_group)

            split_mass = np.sum(action.common_split[group]) + 1e-12

            for user in group:
                local_split = action.common_split[user] / split_mass
                grouped_c_users[user] = local_split * rc_group

        # Private rates are decoded the same way: common stream removed, other private streams interfere.
        for user in range(self.K):
            desired = np.abs(self._project(h[user], private_precoders[user])) ** 2

            interference = 0.0
            for stream in range(self.K):
                if stream != user:
                    interference += (
                        np.abs(self._project(h[user], private_precoders[stream])) ** 2
                    )

            grouped_rp_users[user] = self._rate_from_sinr(
                desired / (interference + self.noise_power)
            )

        grouped_r_users = grouped_c_users + grouped_rp_users

        return grouped_r_users, grouped_common_rates

    def evaluate_action(
        self,
        h: np.ndarray,
        h_hat: np.ndarray,
        action: RSMAAction,
    ) -> TrialResult:
        action.validate(self.K)

        private_dirs = self._build_private_dirs(h_hat)

        (
            rc_users,
            rc,
            c_users,
            rp_users,
            r_users,
            sinr_common,
            sinr_private,
        ) = self._evaluate_global_rsma(h, h_hat, private_dirs, action)

        sum_rate = float(np.sum(r_users))
        fairness = self._jains_fairness(r_users)
        reward = self._reward(r_users)

        sdma_rates = self._evaluate_sdma(h, private_dirs, action)
        sdma_sum_rate = float(np.sum(sdma_rates))
        sdma_fairness = self._jains_fairness(sdma_rates)
        sdma_reward = self._reward(sdma_rates)

        grouped_r_users, grouped_common_rates = self._evaluate_grouped_rsma(
            h, private_dirs, action
        )
        grouped_sum_rate = float(np.sum(grouped_r_users))
        grouped_fairness = self._jains_fairness(grouped_r_users)
        grouped_reward = self._reward(grouped_r_users)

        return TrialResult(
            rc_users=rc_users,
            rc=rc,
            c_users=c_users,
            rp_users=rp_users,
            r_users=r_users,
            sum_rate=sum_rate,
            fairness=fairness,
            reward=reward,
            sdma_rates=sdma_rates,
            sdma_sum_rate=sdma_sum_rate,
            sdma_fairness=sdma_fairness,
            sdma_reward=sdma_reward,
            grouped_r_users=grouped_r_users,
            grouped_sum_rate=grouped_sum_rate,
            grouped_fairness=grouped_fairness,
            grouped_reward=grouped_reward,
            grouped_common_rates=grouped_common_rates,
            sinr_common=sinr_common,
            sinr_private=sinr_private,
            channel_norms=np.linalg.norm(h_hat, axis=1),
            channel_alignment=self._channel_alignment_matrix(h_hat),
        )

    def _rsma_trial(self, action: Optional[RSMAAction] = None) -> TrialResult:
        state = self.sample_state()
        chosen_action = self.default_action() if action is None else action

        return self.evaluate_action(
            h=state["h"],
            h_hat=state["h_hat"],
            action=chosen_action,
        )

    def run(self, action: Optional[RSMAAction] = None) -> Dict[str, float]:
        results = [self._rsma_trial(action=action) for _ in range(self.cfg.num_trials)]

        r_users = np.array([r.r_users for r in results])
        grouped_r_users = np.array([r.grouped_r_users for r in results])
        sdma_rates = np.array([r.sdma_rates for r in results])

        summary = {
            "n_users": self.K,
            "groups": str(self.groups),
            "snr_db": self.cfg.snr_db,
            "num_trials": self.cfg.num_trials,
            "lambda_sum_rate": self.cfg.lambda_sum_rate,

            "avg_global_rsma_sum_rate": float(np.mean([r.sum_rate for r in results])),
            "avg_global_rsma_fairness": float(np.mean([r.fairness for r in results])),
            "avg_global_rsma_reward": float(np.mean([r.reward for r in results])),

            "avg_grouped_rsma_sum_rate": float(np.mean([r.grouped_sum_rate for r in results])),
            "avg_grouped_rsma_fairness": float(np.mean([r.grouped_fairness for r in results])),
            "avg_grouped_rsma_reward": float(np.mean([r.grouped_reward for r in results])),

            "avg_sdma_sum_rate": float(np.mean([r.sdma_sum_rate for r in results])),
            "avg_sdma_fairness": float(np.mean([r.sdma_fairness for r in results])),
            "avg_sdma_reward": float(np.mean([r.sdma_reward for r in results])),

            "grouped_minus_global_sum_rate": float(
                np.mean([r.grouped_sum_rate - r.sum_rate for r in results])
            ),
            "grouped_minus_sdma_sum_rate": float(
                np.mean([r.grouped_sum_rate - r.sdma_sum_rate for r in results])
            ),
            "global_minus_sdma_sum_rate": float(
                np.mean([r.sum_rate - r.sdma_sum_rate for r in results])
            ),

            "avg_global_common_bottleneck_rate": float(np.mean([r.rc for r in results])),
            "avg_group_common_bottleneck_rate": float(
                np.mean([np.mean(r.grouped_common_rates) for r in results])
            ),

            "avg_global_min_user_rate": float(np.mean(np.min(r_users, axis=1))),
            "avg_grouped_min_user_rate": float(np.mean(np.min(grouped_r_users, axis=1))),
            "avg_sdma_min_user_rate": float(np.mean(np.min(sdma_rates, axis=1))),
        }

        for k in range(self.K):
            summary[f"avg_global_total_rate_user_{k}"] = float(np.mean(r_users[:, k]))
            summary[f"avg_grouped_total_rate_user_{k}"] = float(np.mean(grouped_r_users[:, k]))
            summary[f"avg_sdma_rate_user_{k}"] = float(np.mean(sdma_rates[:, k]))

        return summary

    def single_debug_trial(self, action: Optional[RSMAAction] = None) -> Dict[str, object]:
        r = self._rsma_trial(action=action)

        return {
            "global_rc_users": r.rc_users,
            "global_rc": r.rc,
            "global_c_users": r.c_users,
            "global_r_users": r.r_users,
            "global_sum_rate": r.sum_rate,
            "global_fairness": r.fairness,
            "global_reward": r.reward,

            "grouped_common_rates": r.grouped_common_rates,
            "grouped_r_users": r.grouped_r_users,
            "grouped_sum_rate": r.grouped_sum_rate,
            "grouped_fairness": r.grouped_fairness,
            "grouped_reward": r.grouped_reward,

            "sdma_rates": r.sdma_rates,
            "sdma_sum_rate": r.sdma_sum_rate,
            "sdma_fairness": r.sdma_fairness,
            "sdma_reward": r.sdma_reward,

            "sinr_common": r.sinr_common,
            "sinr_private": r.sinr_private,
            "channel_norms": r.channel_norms,
            "channel_alignment": r.channel_alignment,
        }

def print_summary(summary: Dict[str, object]) -> None:
    print(
        f"\nK={summary['n_users']} | groups={summary['groups']} | "
        f"SNR={summary['snr_db']:.1f} dB\n"
        f"Global  : sum={summary['avg_global_rsma_sum_rate']:.3f}, "
        f"fair={summary['avg_global_rsma_fairness']:.3f}, "
        f"reward={summary['avg_global_rsma_reward']:.3f}\n"
        f"Grouped : sum={summary['avg_grouped_rsma_sum_rate']:.3f}, "
        f"fair={summary['avg_grouped_rsma_fairness']:.3f}, "
        f"reward={summary['avg_grouped_rsma_reward']:.3f}\n"
        f"SDMA    : sum={summary['avg_sdma_sum_rate']:.3f}, "
        f"fair={summary['avg_sdma_fairness']:.3f}, "
        f"reward={summary['avg_sdma_reward']:.3f}\n"
        f"d(group-global)={summary['grouped_minus_global_sum_rate']:+.3f} | "
        f"d(group-sdma)={summary['grouped_minus_sdma_sum_rate']:+.3f}"
    )
def run_example() -> None:
    K = 5

    cfg = SimulationConfig(
        n_tx=4,
        n_users=K,
        snr_db=10.0,
        num_trials=5000,
        rng_seed=42,
        csit_error_var=0.05,
        pathloss=np.linspace(1.0, 8.0, K),
        lambda_sum_rate=0.7,
    )

    groups = [
        [0, 1],
        [2, 3, 4],
    ]

    sim = RSMASimulator(cfg, groups=groups)

    action = RSMAAction(
        power_common=0.25,
        power_private=np.full(K, 0.75 / K),
        common_split=np.full(K, 1.0 / K),
    )

    debug = sim.single_debug_trial(action=action)
    summary = sim.run(action=action)
    print_summary(summary)


if __name__ == "__main__":
    run_example()