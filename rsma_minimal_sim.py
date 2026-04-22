import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class SimulationConfig:
    """Configuration for a minimal 2-user downlink MISO RSMA simulation.

    This simulator is intentionally narrow in scope. It is designed to get the
    RSMA logic correct, not to be a full PHY stack.
    """

    n_tx: int = 4
    snr_db: float = 10.0
    num_trials: int = 1000
    rng_seed: int = 7

    # Default RSMA action used by run() and single_debug_trial() unless overridden.
    power_common: float = 0.2
    power_private_1: float = 0.4
    power_private_2: float = 0.4
    common_split_alpha: float = 0.5

    # Imperfect CSIT model: h_hat = h + e, where e ~ CN(0, csit_error_var I)
    csit_error_var: float = 0.0

    # Optional large-scale imbalance between users. Higher pathloss means weaker user.
    pathloss_user_1: float = 1.0
    pathloss_user_2: float = 1.0


@dataclass
class RSMAAction:
    """Decision variables for one RSMA transmission step.

    Parameters
    ----------
    power_common : float
        Fraction of total transmit power allocated to the common stream.
    power_private_1 : float
        Fraction of total transmit power allocated to user 1's private stream.
    power_private_2 : float
        Fraction of total transmit power allocated to user 2's private stream.
    common_split_alpha : float
        Fraction of the common bottleneck rate Rc assigned to user 1.
        User 2 receives the remaining (1-alpha) * Rc.
    """

    power_common: float
    power_private_1: float
    power_private_2: float
    common_split_alpha: float = 0.5

    def validate(self) -> None:
        powers = [self.power_common, self.power_private_1, self.power_private_2]
        if min(powers) < 0:
            raise ValueError("RSMAAction powers must be non-negative")

        total = self.power_common + self.power_private_1 + self.power_private_2
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(
                f"RSMAAction powers must sum to 1.0, got {total:.6f}"
            )

        if not (0.0 <= self.common_split_alpha <= 1.0):
            raise ValueError("common_split_alpha must lie in [0, 1]")


@dataclass
class TrialResult:
    rc1: float
    rc2: float
    rc: float
    rp1: float
    rp2: float
    c1: float
    c2: float
    r1: float
    r2: float
    sum_rate: float
    sdm_rate_1: float
    sdm_rate_2: float
    sdm_sum_rate: float


class RSMASimulator:
    """Minimal 2-user downlink MISO RSMA simulator.

    Signal model:
        x = p_c s_c + p_1 s_1 + p_2 s_2

    User k first decodes the common stream while treating both private streams
    as interference. Then, after perfect SIC of the common stream, user k decodes
    its own private stream while treating the other user's private stream as
    interference.

    Common-rate bottleneck:
        R_c = min(R_c1, R_c2)

    Total user rates:
        R_1 = C_1 + R_p1
        R_2 = C_2 + R_p2
        subject to C_1 + C_2 <= R_c

    Precoding strategy:
        - Private streams: MRT on imperfect CSIT estimates
        - Common stream: normalized sum of private beam directions

    The goal is correctness of RSMA bookkeeping and rate logic.
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.rng_seed)
        self._validate_config()
        self.tx_power = self._db_to_linear(self.cfg.snr_db)
        self.noise_power = 1.0

    def _validate_config(self) -> None:
        if self.cfg.n_tx < 1:
            raise ValueError("n_tx must be at least 1")
        if self.cfg.num_trials < 1:
            raise ValueError("num_trials must be at least 1")
        if self.cfg.pathloss_user_1 <= 0 or self.cfg.pathloss_user_2 <= 0:
            raise ValueError("Pathloss factors must be positive")
        if self.cfg.csit_error_var < 0:
            raise ValueError("csit_error_var must be non-negative")

        self.default_action().validate()

    def default_action(self) -> RSMAAction:
        """Return the default RSMA action implied by SimulationConfig."""
        return RSMAAction(
            power_common=self.cfg.power_common,
            power_private_1=self.cfg.power_private_1,
            power_private_2=self.cfg.power_private_2,
            common_split_alpha=self.cfg.common_split_alpha,
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
            raise ValueError("Cannot normalize a near-zero vector")
        return vec / norm

    def _generate_channels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 2-user flat-fading downlink channels.

        Channel convention:
            h_k is an n_tx-length complex vector.

        Effective received scalar at user k is h_k^H x.
        """
        h1 = self._complex_gaussian_vector(self.cfg.n_tx) / np.sqrt(self.cfg.pathloss_user_1)
        h2 = self._complex_gaussian_vector(self.cfg.n_tx) / np.sqrt(self.cfg.pathloss_user_2)
        return h1, h2

    def _estimate_channels(
        self, h1: np.ndarray, h2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.csit_error_var == 0.0:
            return h1.copy(), h2.copy()

        sigma = np.sqrt(self.cfg.csit_error_var)
        e1 = sigma * self._complex_gaussian_vector(self.cfg.n_tx)
        e2 = sigma * self._complex_gaussian_vector(self.cfg.n_tx)
        return h1 + e1, h2 + e2

    def _build_precoders(
        self, h1_hat: np.ndarray, h2_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct simple beamformers from CSIT estimates.

        Private beamformers: MRT
        Common beamformer: normalized sum of private directions
        """
        p1_dir = self._normalize(h1_hat)
        p2_dir = self._normalize(h2_hat)
        pc_dir = self._normalize(p1_dir + p2_dir)
        return pc_dir, p1_dir, p2_dir

    def _project(self, h: np.ndarray, p: np.ndarray) -> complex:
        return np.vdot(h, p)  # h^H p

    def _rate_from_sinr(self, sinr: float) -> float:
        return float(np.log2(1.0 + max(sinr, 0.0)))

    def sample_state(self) -> Dict[str, np.ndarray]:
        """Sample one channel state for evaluation."""
        h1, h2 = self._generate_channels()
        h1_hat, h2_hat = self._estimate_channels(h1, h2)
        return {
            "h1": h1,
            "h2": h2,
            "h1_hat": h1_hat,
            "h2_hat": h2_hat,
        }

    def evaluate_action(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        h1_hat: np.ndarray,
        h2_hat: np.ndarray,
        action: RSMAAction,
    ) -> TrialResult:
        """Evaluate one RSMA action on a provided channel state."""
        action.validate()

        pc_dir, p1_dir, p2_dir = self._build_precoders(h1_hat, h2_hat)

        # Power-scaled precoders. Total transmit power is tx_power.
        pc = np.sqrt(self.tx_power * action.power_common) * pc_dir
        p1 = np.sqrt(self.tx_power * action.power_private_1) * p1_dir
        p2 = np.sqrt(self.tx_power * action.power_private_2) * p2_dir

        # Effective channel gains for each stream at each user.
        g1c = np.abs(self._project(h1, pc)) ** 2
        g11 = np.abs(self._project(h1, p1)) ** 2
        g12 = np.abs(self._project(h1, p2)) ** 2

        g2c = np.abs(self._project(h2, pc)) ** 2
        g21 = np.abs(self._project(h2, p1)) ** 2
        g22 = np.abs(self._project(h2, p2)) ** 2

        # Decode common stream first: both private streams are interference.
        sinr_c1 = g1c / (g11 + g12 + self.noise_power)
        sinr_c2 = g2c / (g21 + g22 + self.noise_power)
        rc1 = self._rate_from_sinr(sinr_c1)
        rc2 = self._rate_from_sinr(sinr_c2)
        rc = min(rc1, rc2)

        # Common-rate split is now action-driven.
        c1 = action.common_split_alpha * rc
        c2 = (1.0 - action.common_split_alpha) * rc

        # After perfect SIC of the common stream, each user decodes its own private stream.
        sinr_p1 = g11 / (g12 + self.noise_power)
        sinr_p2 = g22 / (g21 + self.noise_power)
        rp1 = self._rate_from_sinr(sinr_p1)
        rp2 = self._rate_from_sinr(sinr_p2)

        r1 = c1 + rp1
        r2 = c2 + rp2
        sum_rate = r1 + r2

        # SDMA baseline using same private beam directions and private powers renormalized.
        private_power_total = max(action.power_private_1 + action.power_private_2, 1e-12)
        sdma_p1_power = self.tx_power * action.power_private_1 / private_power_total
        sdma_p2_power = self.tx_power * action.power_private_2 / private_power_total
        p1_sdma = np.sqrt(sdma_p1_power) * p1_dir
        p2_sdma = np.sqrt(sdma_p2_power) * p2_dir

        s11 = np.abs(self._project(h1, p1_sdma)) ** 2
        s12 = np.abs(self._project(h1, p2_sdma)) ** 2
        s21 = np.abs(self._project(h2, p1_sdma)) ** 2
        s22 = np.abs(self._project(h2, p2_sdma)) ** 2

        sinr_sdma_1 = s11 / (s12 + self.noise_power)
        sinr_sdma_2 = s22 / (s21 + self.noise_power)
        sdm_rate_1 = self._rate_from_sinr(sinr_sdma_1)
        sdm_rate_2 = self._rate_from_sinr(sinr_sdma_2)
        sdm_sum_rate = sdm_rate_1 + sdm_rate_2

        return TrialResult(
            rc1=rc1,
            rc2=rc2,
            rc=rc,
            rp1=rp1,
            rp2=rp2,
            c1=c1,
            c2=c2,
            r1=r1,
            r2=r2,
            sum_rate=sum_rate,
            sdm_rate_1=sdm_rate_1,
            sdm_rate_2=sdm_rate_2,
            sdm_sum_rate=sdm_sum_rate,
        )

    def _rsma_trial(self, action: Optional[RSMAAction] = None) -> TrialResult:
        """Run one sampled trial using the provided action or the default one."""
        state = self.sample_state()
        chosen_action = self.default_action() if action is None else action
        return self.evaluate_action(
            h1=state["h1"],
            h2=state["h2"],
            h1_hat=state["h1_hat"],
            h2_hat=state["h2_hat"],
            action=chosen_action,
        )

    def run(self, action: Optional[RSMAAction] = None) -> Dict[str, float]:
        results = [self._rsma_trial(action=action) for _ in range(self.cfg.num_trials)]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(r, attr) for r in results]))

        summary = {
            "snr_db": self.cfg.snr_db,
            "num_trials": self.cfg.num_trials,
            "avg_common_rate_user_1": mean("rc1"),
            "avg_common_rate_user_2": mean("rc2"),
            "avg_common_bottleneck_rate": mean("rc"),
            "avg_private_rate_user_1": mean("rp1"),
            "avg_private_rate_user_2": mean("rp2"),
            "avg_total_rate_user_1": mean("r1"),
            "avg_total_rate_user_2": mean("r2"),
            "avg_rsma_sum_rate": mean("sum_rate"),
            "avg_sdma_rate_user_1": mean("sdm_rate_1"),
            "avg_sdma_rate_user_2": mean("sdm_rate_2"),
            "avg_sdma_sum_rate": mean("sdm_sum_rate"),
            "rsma_minus_sdma_sum_rate": mean("sum_rate") - mean("sdm_sum_rate"),
        }
        return summary

    def single_debug_trial(self, action: Optional[RSMAAction] = None) -> Dict[str, float]:
        """Run one trial and expose detailed rates for sanity checks."""
        r = self._rsma_trial(action=action)
        return {
            "rc1": r.rc1,
            "rc2": r.rc2,
            "rc": r.rc,
            "c1": r.c1,
            "c2": r.c2,
            "rp1": r.rp1,
            "rp2": r.rp2,
            "r1": r.r1,
            "r2": r.r2,
            "sum_rate": r.sum_rate,
            "sdm_rate_1": r.sdm_rate_1,
            "sdm_rate_2": r.sdm_rate_2,
            "sdm_sum_rate": r.sdm_sum_rate,
        }


def run_example() -> None:
    cfg = SimulationConfig(
        n_tx=4,
        snr_db=10.0,
        num_trials=5000,
        rng_seed=42,
        power_common=0.2,
        power_private_1=0.4,
        power_private_2=0.4,
        common_split_alpha=0.5,
        csit_error_var=0.05,
        pathloss_user_1=1.0,
        pathloss_user_2=2.0,
    )

    sim = RSMASimulator(cfg)

    # You can either rely on the default action embedded in cfg,
    # or pass an explicit RSMAAction here.
    action = RSMAAction(
        power_common=0.2,
        power_private_1=0.4,
        power_private_2=0.4,
        common_split_alpha=0.5,
    )

    debug = sim.single_debug_trial(action=action)
    summary = sim.run(action=action)

    print("Single-trial sanity check:")
    for k, v in debug.items():
        print(f"  {k:>24s}: {v:.4f}")

    print("\nAveraged summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>24s}: {v:.4f}")
        else:
            print(f"  {k:>24s}: {v}")


if __name__ == "__main__":
    run_example()
