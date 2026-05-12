import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class SimulationConfig:
    """Generic K-user downlink MISO RSMA simulation config."""

    n_tx: int = 4
    n_users: int = 2
    snr_db: float = 10.0
    num_trials: int = 1000
    rng_seed: int = 7

    # Imperfect CSIT: h_hat = h + e
    csit_error_var: float = 0.0

    # Optional per-user pathloss. If None, all users use pathloss = 1.
    pathloss: Optional[np.ndarray] = None


@dataclass
class RSMAAction:
    """K-user RSMA action.

    power_common:
        Fraction of total transmit power assigned to common stream.

    power_private:
        Length-K vector of private-stream power fractions.

    common_split:
        Length-K vector assigning fractions of common bottleneck rate.
    """

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

    sdma_rates: np.ndarray
    sdma_sum_rate: float

    sinr_common: np.ndarray
    sinr_private: np.ndarray
    channel_norms: np.ndarray
    channel_alignment: np.ndarray


class RSMASimulator:
    """Generic K-user downlink MISO RSMA simulator.

    Signal model:
        x = p_c s_c + sum_k p_k s_k

    Each user decodes:
        1. common stream, treating all private streams as interference
        2. its own private stream after perfect common-stream SIC

    This is still a single-layer common-stream RSMA model.
    """

    def __init__(self, config: SimulationConfig):
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

    def _validate_config(self) -> None:
        if self.cfg.n_tx < 1:
            raise ValueError("n_tx must be at least 1")

        if self.cfg.n_users < 2:
            raise ValueError("n_users must be at least 2")

        if self.cfg.num_trials < 1:
            raise ValueError("num_trials must be at least 1")

        if self.cfg.csit_error_var < 0:
            raise ValueError("csit_error_var must be non-negative")

        if self.cfg.pathloss is not None:
            pathloss = np.asarray(self.cfg.pathloss, dtype=float)
            if len(pathloss) != self.cfg.n_users:
                raise ValueError("pathloss must have length n_users")
            if np.any(pathloss <= 0):
                raise ValueError("all pathloss values must be positive")

    def default_action(self) -> RSMAAction:
        """Uniform default action."""
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

    def _project(self, h: np.ndarray, p: np.ndarray) -> complex:
        return np.vdot(h, p)

    def sample_state(self) -> Dict[str, np.ndarray]:
        """Sample true and estimated K-user channel state."""

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

        return {
            "h": h,
            "h_hat": h_hat,
        }

    def _build_precoders(self, h_hat: np.ndarray) -> Dict[str, np.ndarray]:
        """Build common and private precoder directions.

        Private streams:
            MRT for each user.

        Common stream:
            normalized sum of all private beam directions.
        """

        private_dirs = np.zeros_like(h_hat, dtype=complex)

        for k in range(self.K):
            private_dirs[k] = self._normalize(h_hat[k])

        common_dir = self._normalize(np.sum(private_dirs, axis=0))

        return {
            "common_dir": common_dir,
            "private_dirs": private_dirs,
        }

    def _channel_alignment_matrix(self, h_hat: np.ndarray) -> np.ndarray:
        """Return K x K normalized channel alignment matrix."""

        align = np.zeros((self.K, self.K))

        norms = np.linalg.norm(h_hat, axis=1)

        for i in range(self.K):
            for j in range(self.K):
                denom = norms[i] * norms[j] + 1e-12
                align[i, j] = np.abs(np.vdot(h_hat[i], h_hat[j])) / denom

        return align

    def evaluate_action(
        self,
        h: np.ndarray,
        h_hat: np.ndarray,
        action: RSMAAction,
    ) -> TrialResult:
        """Evaluate one K-user RSMA action on one channel state."""

        action.validate(self.K)

        precoders = self._build_precoders(h_hat)
        common_dir = precoders["common_dir"]
        private_dirs = precoders["private_dirs"]

        pc = np.sqrt(self.tx_power * action.power_common) * common_dir

        private_precoders = np.zeros_like(private_dirs, dtype=complex)
        for k in range(self.K):
            private_precoders[k] = (
                np.sqrt(self.tx_power * action.power_private[k]) * private_dirs[k]
            )

        # gain_private[user, stream] = |h_user^H p_stream|^2
        gain_private = np.zeros((self.K, self.K))
        gain_common = np.zeros(self.K)

        for user in range(self.K):
            gain_common[user] = np.abs(self._project(h[user], pc)) ** 2

            for stream in range(self.K):
                gain_private[user, stream] = (
                    np.abs(self._project(h[user], private_precoders[stream])) ** 2
                )

        # Common decoding: all private streams are interference.
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

        # Private decoding after perfect common SIC.
        sinr_private = np.zeros(self.K)
        rp_users = np.zeros(self.K)

        for user in range(self.K):
            desired = gain_private[user, user]
            interference = np.sum(gain_private[user, :]) - desired

            sinr_private[user] = desired / (interference + self.noise_power)
            rp_users[user] = self._rate_from_sinr(sinr_private[user])

        r_users = c_users + rp_users
        sum_rate = float(np.sum(r_users))

        # SDMA baseline: no common stream, private powers renormalized.
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

        sdma_sum_rate = float(np.sum(sdma_rates))

        return TrialResult(
            rc_users=rc_users,
            rc=rc,
            c_users=c_users,
            rp_users=rp_users,
            r_users=r_users,
            sum_rate=sum_rate,
            sdma_rates=sdma_rates,
            sdma_sum_rate=sdma_sum_rate,
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

        rsma_sum_rates = np.array([r.sum_rate for r in results])
        sdma_sum_rates = np.array([r.sdma_sum_rate for r in results])

        r_users = np.array([r.r_users for r in results])
        rp_users = np.array([r.rp_users for r in results])
        c_users = np.array([r.c_users for r in results])
        rc_users = np.array([r.rc_users for r in results])

        summary = {
            "n_users": self.K,
            "snr_db": self.cfg.snr_db,
            "num_trials": self.cfg.num_trials,
            "avg_rsma_sum_rate": float(np.mean(rsma_sum_rates)),
            "avg_sdma_sum_rate": float(np.mean(sdma_sum_rates)),
            "rsma_minus_sdma_sum_rate": float(
                np.mean(rsma_sum_rates) - np.mean(sdma_sum_rates)
            ),
            "avg_common_bottleneck_rate": float(np.mean([r.rc for r in results])),
            "avg_min_user_rate": float(np.mean(np.min(r_users, axis=1))),
            "avg_user_rate_std": float(np.mean(np.std(r_users, axis=1))),
        }

        for k in range(self.K):
            summary[f"avg_total_rate_user_{k}"] = float(np.mean(r_users[:, k]))
            summary[f"avg_private_rate_user_{k}"] = float(np.mean(rp_users[:, k]))
            summary[f"avg_common_alloc_user_{k}"] = float(np.mean(c_users[:, k]))
            summary[f"avg_common_decodable_user_{k}"] = float(np.mean(rc_users[:, k]))

        return summary

    def single_debug_trial(self, action: Optional[RSMAAction] = None) -> Dict[str, object]:
        r = self._rsma_trial(action=action)

        return {
            "rc_users": r.rc_users,
            "rc": r.rc,
            "c_users": r.c_users,
            "rp_users": r.rp_users,
            "r_users": r.r_users,
            "sum_rate": r.sum_rate,
            "sdma_rates": r.sdma_rates,
            "sdma_sum_rate": r.sdma_sum_rate,
            "sinr_common": r.sinr_common,
            "sinr_private": r.sinr_private,
            "channel_norms": r.channel_norms,
            "channel_alignment": r.channel_alignment,
        }


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
    )

    sim = RSMASimulator(cfg)

    action = RSMAAction(
        power_common=0.25,
        power_private=np.full(K, 0.75 / K),
        common_split=np.full(K, 1.0 / K),
    )

    debug = sim.single_debug_trial(action=action)
    summary = sim.run(action=action)

    print("Single-trial sanity check:")
    for k, v in debug.items():
        print(f"{k}: {v}")

    print("\nAveraged summary:")
    for k, v in summary.items():
        print(f"{k:>32s}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    run_example()