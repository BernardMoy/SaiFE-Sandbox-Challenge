"""
CMA-ES training for The Concentrator (time-boxed to ~40 minutes).

Policy: tiny numpy MLP (5 -> HIDDEN -> 3), tanh activations.
Inputs (per trajectory):
    0: mispricing_norm = (midprice - sqrt_price**2) / midprice
    1: lower_dist       = (current_tick - lp_tick_lower) / tau
    2: upper_dist       = (lp_tick_upper - current_tick) / tau
    3: time_left        = 1 - time / terminal_time
    4: gas_frac         = clip(gas_cost / portfolio_value, 0, 10)

Outputs -> action:
    lower_offset = tau * tanh(out[0])
    upper_offset = tau * tanh(out[1])
    hold_flag    = tanh(out[2])   (forced to -1 until lp_ever_deployed)

Fitness = mean(candidate_final_wealth - HoldToken0_final_wealth), evaluated
with common random numbers (same seed for candidate and baseline) to cancel
path noise and directly target "beat the baseline".

Usage:
    python experiments/train_cma_es_agent.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cma

from SAiFE_gym.challenge import ScenarioConfig, create_environment
from SAiFE_gym.agents.BaselineAgents import HoldToken0Agent
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    POOL_SQRT_PRICE_KEY,
    POOL_CURRENT_TICK_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_EVER_DEPLOYED_KEY,
    TIME_KEY,
    GAS_COST_KEY,
    PORTFOLIO_VALUE_KEY,
)

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
IN_DIM = 5
HIDDEN = 8
OUT_DIM = 3
N_PARAMS = IN_DIM * HIDDEN + HIDDEN + HIDDEN * OUT_DIM + OUT_DIM  # 75

_OFFICIAL = ScenarioConfig()
TAU = float(_OFFICIAL.tau)
TERMINAL_TIME = float(_OFFICIAL.terminal_time)
N_STEPS = _OFFICIAL.n_steps

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
TRAIN_SEEDS = list(range(10000, 10016))
VAL_SEEDS = [42, 314, 2718]
TRAIN_NTRAJ = 20
VAL_NTRAJ = 100
POPSIZE = 8
BATCH_SEEDS = 2
TIME_BUDGET_SECONDS = 35 * 60
VAL_EVERY = 15
OUT_PATH = os.path.join(os.path.dirname(__file__), "cma_best_params.npy")


def unpack(params):
    params = np.asarray(params, dtype=np.float64)
    i = 0
    W1 = params[i:i + IN_DIM * HIDDEN].reshape(IN_DIM, HIDDEN); i += IN_DIM * HIDDEN
    b1 = params[i:i + HIDDEN]; i += HIDDEN
    W2 = params[i:i + HIDDEN * OUT_DIM].reshape(HIDDEN, OUT_DIM); i += HIDDEN * OUT_DIM
    b2 = params[i:i + OUT_DIM]; i += OUT_DIM
    return W1, b1, W2, b2


def features(state):
    midprice = state[ASSET_PRICE_KEY]
    sqrt_price = state[POOL_SQRT_PRICE_KEY]
    current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.float64)
    lp_lower = state[LP_TICK_LOWER_KEY].astype(np.float64)
    lp_upper = state[LP_TICK_UPPER_KEY].astype(np.float64)
    t = state[TIME_KEY]
    gas = state[GAS_COST_KEY]
    pv = state[PORTFOLIO_VALUE_KEY]

    mispricing_norm = (midprice - sqrt_price ** 2) / midprice
    lower_dist = (current_tick - lp_lower) / TAU
    upper_dist = (lp_upper - current_tick) / TAU
    time_left = 1.0 - t / TERMINAL_TIME
    gas_frac = np.clip(gas / np.maximum(pv, 1e-3), 0.0, 10.0)

    return np.column_stack([mispricing_norm, lower_dist, upper_dist, time_left, gas_frac])


def policy_action(params, state):
    W1, b1, W2, b2 = unpack(params)
    x = features(state)
    h = np.tanh(x @ W1 + b1)
    out = np.tanh(h @ W2 + b2)

    lower = TAU * out[:, 0]
    upper = TAU * out[:, 1]
    hold = out[:, 2]

    ever_deployed = state[LP_EVER_DEPLOYED_KEY]
    hold = np.where(ever_deployed, hold, -1.0)
    return np.column_stack([lower, upper, hold])


def rollout_final_wealth(params, seed, num_trajectories, holdtoken0=False):
    cfg = ScenarioConfig(num_trajectories=num_trajectories)
    env = create_environment(cfg, seed=seed)
    obs, _ = env.reset()
    agent = HoldToken0Agent(env) if holdtoken0 else None

    cum = np.zeros(num_trajectories)
    for _ in range(env.n_steps):
        if holdtoken0:
            action = agent.get_action(obs)
        else:
            action = policy_action(params, obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        cum += rewards
    return cfg.initial_wealth + cum


def eval_candidate_on_seed(args):
    params, seed, num_trajectories, baseline_wealth = args
    candidate_wealth = rollout_final_wealth(params, seed, num_trajectories)
    return float(np.mean(candidate_wealth - baseline_wealth))


def precompute_baseline(seeds, num_trajectories):
    baseline = {}
    for s in seeds:
        baseline[s] = rollout_final_wealth(None, s, num_trajectories, holdtoken0=True)
    return baseline


def validate(params, val_baseline):
    diffs = []
    for s in VAL_SEEDS:
        w = rollout_final_wealth(params, s, VAL_NTRAJ)
        diffs.append(np.mean(w - val_baseline[s]))
    return float(np.mean(diffs))


def main():
    from multiprocessing import Pool

    print(f"N_PARAMS = {N_PARAMS}")
    print("Precomputing HoldToken0 baseline wealth for train/val seeds...")
    t0 = time.time()
    train_baseline = precompute_baseline(TRAIN_SEEDS, TRAIN_NTRAJ)
    val_baseline = precompute_baseline(VAL_SEEDS, VAL_NTRAJ)
    print(f"  done in {time.time() - t0:.1f}s")

    # Domain-informed init: approximate DeployOnceWideAgent
    # (full-range deploy on first step, then hold forever).
    x0 = np.zeros(N_PARAMS)
    x0[-3:] = [-2.0, 2.0, 2.0]  # b2 -> lower~-tau, upper~+tau, hold>0

    es = cma.CMAEvolutionStrategy(x0, 0.5, {"popsize": POPSIZE, "seed": 1, "verbose": -9})

    rng = np.random.default_rng(0)
    best_val_score = -np.inf
    best_params = x0.copy()

    n_workers = min(POPSIZE * BATCH_SEEDS, os.cpu_count() or 1)
    print(f"Using {n_workers} worker processes")

    start = time.time()
    gen = 0
    with Pool(n_workers) as pool:
        while time.time() - start < TIME_BUDGET_SECONDS:
            gen += 1
            solutions = es.ask()
            batch_seeds = rng.choice(TRAIN_SEEDS, size=BATCH_SEEDS, replace=False)

            tasks = [
                (sol, int(s), TRAIN_NTRAJ, train_baseline[int(s)])
                for sol in solutions
                for s in batch_seeds
            ]
            results = pool.map(eval_candidate_on_seed, tasks)
            results = np.asarray(results).reshape(POPSIZE, BATCH_SEEDS)
            fitness = results.mean(axis=1)

            # CMA-ES minimizes -> negate to maximize fitness (excess PnL over baseline)
            es.tell(solutions, (-fitness).tolist())

            elapsed = time.time() - start
            print(f"gen {gen:4d}  elapsed {elapsed:6.1f}s  mean_fit {fitness.mean():+8.4f}  best_fit {fitness.max():+8.4f}")

            if gen % VAL_EVERY == 0 or elapsed >= TIME_BUDGET_SECONDS:
                val_score = validate(es.mean, val_baseline)
                print(f"  [val] gen {gen}  mean excess PnL over HoldToken0 (practice seeds) = {val_score:+.4f}")
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_params = np.array(es.mean, copy=True)
                    np.save(OUT_PATH, best_params)
                    print(f"  -> new best, saved to {OUT_PATH}")

    print(f"\nDone. Best validation excess PnL over HoldToken0: {best_val_score:+.4f}")
    np.save(OUT_PATH, best_params)
    print(f"Saved best params to {OUT_PATH}")


if __name__ == "__main__":
    main()
