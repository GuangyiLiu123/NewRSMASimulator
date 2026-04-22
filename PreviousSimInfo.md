# RSMA–SDMA RT Simulation (Sionna-Based)

## Overview

This script implements a **hybrid wireless simulation pipeline** that combines:

- Ray-traced large-scale propagation (via Sionna RT)
- Synthetic pseudo-vector MISO channels
- Analytical RSMA vs SDMA rate evaluation

It is designed to study **RSMA behavior under realistic spatial conditions** without building a full PHY-layer simulator.

---

## What This Code Does

### 1. Ray-Traced Environment (Macro Geometry)

- Loads a real-world city scene (Florence)
- Places multiple transmitters (base stations)
- Computes a **radio map (path gain)**
- Samples:
  - one fixed UE (UE1)
  - many candidate UEs (UE2)
- Extracts:
  - direct gains (`G1`, `G2`)
  - inter-cell interference (`I1_ext`, `I2_ext`)

---

### 2. Pseudo-Vector Channel Construction

Converts scalar gains into **complex MISO channel vectors**:

- UE1 → single vector `h1`
- UE2 → multiple vectors `h2[i]`
- Controls **channel alignment** via `alignment_mix`
  - higher alignment → harder SDMA regime
- Ensures:
  - ‖h‖² ≈ path gain

---

### 3. Imperfect CSIT Modeling

Adds estimation noise:

- `h_est = h_true + noise`
- Controlled by `nmse_surrogate`

This allows simulation of **beamforming mismatch**, which is critical for RSMA gains.

---

### 4. Effective Gain Computation

Builds beams and computes:

- Private gains (`g11, g22`)
- Interference gains (`g12, g21`)
- Common-stream gains (`g1c, g2c`)

Two modes:
- perfect CSIT
- mismatched CSIT (more realistic)

---

### 5. Rate Computation

#### SDMA
- Standard SINR-based rates
- Treats interference as noise

#### RSMA
- Common stream decoded first
- Bottleneck rate: `Rc = min(Rc1, Rc2)`
- Then private decoding after SIC

---

### 6. RSMA Optimization

- Searches over common power `Pc`
- Uses grid search:
  - `Pc ∈ [0, 0.6]`
  - `P1 = P2 = (1 - Pc)/2`
- Returns:
  - best sum-rate
  - optimal `Pc` per sample

---

### 7. Simulation Outputs

- **Average RSMA vs SDMA rates vs SNR**
- Spatial scatter plots:
  - UE2 performance maps
- City visualization:
  - radio map + UE positions
- Diagnostics:
  - gain statistics
  - optimal power allocation trends

---

## Key Idea

This is **not a full channel simulator**.

It separates:

- **Geometry + propagation** → handled by ray tracing  
- **Spatial structure (MISO)** → approximated via pseudo-vectors  
- **Rate logic** → computed analytically  

---

## Limitations

### 1. Not a True PHY Model
- No OFDM
- No coding/modulation
- No time/frequency selectivity
- No link-level simulation

---

### 2. Synthetic Channel Structure
- Pseudo-vectors are **not physically derived**
- Alignment is artificially controlled
- No spatial correlation modeling beyond mixing

---

### 3. Simplified Beamforming
- MRT only
- No ZF / WMMSE / optimal precoding
- Common beam = simple sum direction

---

### 4. RSMA Optimization is Limited
- Grid search over one parameter (`Pc`)
- No joint optimization of:
  - beamformers
  - user grouping
  - rate splitting

---

### 5. Interference Model is Approximate
- Inter-cell interference treated as scalar power
- No phase-aware multi-cell beam interaction

---

### 6. 2-User Structure
- One fixed UE1 + many UE2 samples
- Not a general multi-user scheduling framework

---

## When This Is Useful

Use this when you want:

- Fast evaluation of **RSMA vs SDMA trends**
- Insight into **when RSMA helps (alignment + CSIT error)**
- A bridge between:
  - toy simulations
  - full RT-based environments

---

## When It Is Not Enough

Do **not** rely on this for:

- publication-level PHY accuracy
- standard-compliant system evaluation
- real beamforming performance claims
- multi-user scheduling studies

---

## Bottom Line

This code is a **controlled experimental environment**:

- realistic geometry  
- simplified channel structure  
- correct RSMA rate logic  

It is best viewed as a **research prototyping tool**, not a final simulator.