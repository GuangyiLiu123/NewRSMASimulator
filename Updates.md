# RSMA Simulator

## Overview

This is a **K-user downlink MISO RSMA simulator** designed to study:

- Global RSMA (single common stream)
- Grouped RSMA (one common stream per group)
- SDMA baseline

The simulator includes:
- Rayleigh fading channels with pathloss imbalance
- Imperfect CSIT
- MRT precoding
- SIC decoding (common → private)
- Static user grouping
- Mixed objective:
  

  
### Interpretation

- **Grouped RSMA significantly outperforms global RSMA**
  - (+0.207 sum-rate gain)
  - Confirms that a single common stream does not scale with user count

- **Grouped RSMA slightly beats SDMA**
  - (+0.033 sum-rate gain)
  - Indicates useful interference management via common streams

- **Fairness tradeoff**
  - SDMA has slightly higher fairness
  - Grouped RSMA prioritizes throughput over uniformity

### Key Insight

> Global common streams collapse due to weakest-user bottlenecks.  
> Grouping restores effectiveness by localizing interference.

---

## Current Limitations

- Static grouping (manually defined)
- Uniform power allocation across users
- Uniform common-rate splitting
- i.i.d. Rayleigh channels (no spatial structure)
- No temporal dynamics or scheduling

---

## Roadmap (Short-Term)

### 1. Structured Channel Models
Introduce meaningful user structure:
- Clustered users (aligned channels)
- Near–far scenarios
- Correlated fading

This is critical for making grouping meaningful.

---

### 2. Group Diagnostics
Add metrics to explain behavior:
- Per-group common bottlenecks
- Intra-group vs inter-group alignment
- Group pathloss statistics

---

### 3. Better Action Space
Extend beyond:
- `power_common`

Add:
- Per-group common power allocation
- Non-uniform private power allocation
- Non-uniform common rate splitting

---

### 4. Group Generation Utilities
Before learning:
- Alignment-based grouping
- Pathloss-based grouping
- Random/group-size-based baselines

---

## On Agentic Methods

Agent-based optimization (bandits / RL) is **temporarily on hold**.

Reason:
- The environment is still too shallow (uniform actions, unstructured channels)
- Meaningful learning requires richer state + action spaces

Next step is to **strengthen the simulator**, not the agent.

---

## Summary

The simulator has reached a key milestone:

- Successfully reproduces the core limitation of global RSMA
- Demonstrates the benefit of grouping
- Shows realistic tradeoffs between throughput and fairness

The next phase is to make grouping **structure-aware**, not learned yet.