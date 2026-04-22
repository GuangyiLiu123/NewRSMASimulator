# RSMA Simulator Documentation

This file documents a small Python simulator for a **2-user downlink MISO RSMA** system. It is meant to validate the **RSMA logic and rate calculations**, not to act as a full wireless PHY simulator.

## What the code does

The simulator models a transmitter with multiple antennas sending:

- one **common stream** for both users
- one **private stream** for user 1
- one **private stream** for user 2

It then computes the achievable rates under a simplified RSMA decoding process:

1. Each user decodes the **common stream** first.
2. While decoding the common stream, both private streams are treated as interference.
3. After removing the common stream through ideal SIC, each user decodes its own private stream.
4. The code compares the RSMA result against an **SDMA baseline**.

## Main classes

### `SimulationConfig`
Stores global simulation settings such as:

- `n_tx`: number of transmit antennas
- `snr_db`: transmit SNR in dB
- `num_trials`: number of Monte Carlo trials
- `rng_seed`: random seed
- default RSMA power split values
- `csit_error_var`: channel estimation error variance
- `pathloss_user_1`, `pathloss_user_2`: large-scale pathloss factors

This class defines the overall environment the simulator will use.

---

### `RSMAAction`
Represents one RSMA decision:

- `power_common`
- `power_private_1`
- `power_private_2`
- `common_split_alpha`

The `validate()` method ensures:

- powers are nonnegative
- the three powers sum to `1.0`
- `common_split_alpha` lies in `[0, 1]`

So this object defines **how transmit power and common rate are split**.

---

### `TrialResult`
Stores the result of one evaluated channel realization.

Important fields include:

- `rc1`, `rc2`: common-stream decoding rates at each user
- `rc`: common bottleneck rate
- `rp1`, `rp2`: private rates
- `c1`, `c2`: assigned common-rate portions
- `r1`, `r2`: total user rates
- `sum_rate`: total RSMA sum rate
- `sdm_sum_rate`: SDMA baseline sum rate

---

### `RSMASimulator`
This is the main simulation engine.

It handles:

- channel generation
- imperfect CSIT generation
- beamformer construction
- SINR and rate calculations
- trial averaging

## Channel and signal model

The code assumes a **2-user flat-fading downlink MISO** model.

The transmitted signal is conceptually:

```text
x = p_c s_c + p_1 s_1 + p_2 s_2
```

where:

- `s_c` is the common stream
- `s_1` and `s_2` are private streams
- `p_c`, `p_1`, `p_2` are the beamformers

The effective received scalar is computed using the Hermitian inner product:

```text
h_k^H x
```

## Precoding strategy

The beamforming is intentionally simple:

- **Private streams** use MRT based on estimated channels
- **Common stream** uses the normalized sum of the two private beam directions

This is not meant to be optimal beamforming. It is just a clean, interpretable baseline.

## Important methods

### `sample_state()`
Generates:

- true channel for user 1 and user 2
- estimated channel for user 1 and user 2

This is useful if you want to inspect or reuse one channel state.

---

### `evaluate_action(h1, h2, h1_hat, h2_hat, action)`
Evaluates one `RSMAAction` on a specific channel realization.

What it computes:

- common-stream SINR and rate at both users
- common bottleneck rate
- common-rate split between users
- private-stream SINR and rate after ideal SIC
- total RSMA user rates and sum rate
- SDMA baseline rates

This is the core function of the simulator.

---

### `single_debug_trial(action=None)`
Runs a single random trial and returns detailed values.

Use this for sanity checks when you want to inspect one example realization.

---

### `run(action=None)`
Runs many trials and returns averaged results.

Returned summary fields include:

- average RSMA common/private/total rates
- average SDMA rates
- `rsma_minus_sdma_sum_rate`

This is the main method for evaluating average performance.

## How RSMA is implemented here

The logic is:

- Compute the common rate each user can decode: `rc1`, `rc2`
- Take the bottleneck:

```text
rc = min(rc1, rc2)
```

- Split that bottleneck rate using `common_split_alpha`
- Add each user's private rate:

```text
r1 = c1 + rp1
r2 = c2 + rp2
```

So the simulator follows the standard RSMA bookkeeping structure.

## SDMA baseline

The code also builds an SDMA comparison using:

- the same private beam directions
- the same private-power proportions, renormalized after removing common-stream power

This gives a simple apples-to-apples baseline, though it is still a simplified one.

## Example workflow

Typical usage looks like this:

```python
cfg = SimulationConfig(num_trials=5000, snr_db=10.0)
sim = RSMASimulator(cfg)

action = RSMAAction(
    power_common=0.2,
    power_private_1=0.4,
    power_private_2=0.4,
    common_split_alpha=0.5,
)

debug = sim.single_debug_trial(action=action)
summary = sim.run(action=action)
```

## What this code is good for

This simulator is good for:

- checking RSMA rate logic
- debugging common/private stream bookkeeping
- testing how power splits affect performance
- comparing RSMA against a simple SDMA baseline
- serving as a foundation for future optimization or RL work

## What it does **not** do

This is **not** a full communications simulator. It does not include:

- OFDM or frequency selectivity
- coding/modulation details
- realistic receiver processing beyond simple SIC assumptions
- advanced beamforming optimization
- more than 2 users
- packet-level or standard-compliant PHY modeling

## Bottom line

This file is a **minimal, clean RSMA simulation environment**. Its main value is that it clearly separates:

- configuration
- action definition
- one-trial evaluation
- averaged simulation results

That makes it a good base for expansion, especially if the next goal is to add:

- better beamforming
- more users
- fairness objectives
- optimization or reinforcement learning
