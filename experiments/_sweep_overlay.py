"""One-off sweep of the heuristic-overlay constants in my_agent.Agent."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from SAiFE_gym.challenge import ScenarioConfig, create_environment
import implementations.cma_es as cma_es

PRACTICE_SEEDS = [42, 314, 2718]


def run_episode(env, agent):
    state, _ = env.reset()
    cum = np.zeros(env.num_trajectories)
    terminated = np.zeros(env.num_trajectories, dtype=bool)
    while not np.any(terminated):
        action = agent.get_action(state)
        state, reward, terminated, _, _ = env.step(action)
        cum += reward
    return cum


def evaluate(skew, breach_ticks, breach_steps, early_scale, late_scale):
    config = ScenarioConfig()
    ns = config.submission_namespace()
    pnls = []
    for seed in PRACTICE_SEEDS:
        env = create_environment(config, seed)
        agent = cma_es.Agent(ns)
        agent.SKEW_TICKS = skew
        agent.UPSIDE_BREACH_TICKS = breach_ticks
        agent.UPSIDE_BREACH_STEPS = breach_steps
        agent.EARLY_WIDTH_SCALE = early_scale
        agent.LATE_WIDTH_SCALE = late_scale
        pnls.append(run_episode(env, agent))
    pnl = np.concatenate(pnls)
    return pnl.mean(), pnl.std(), pnl.min(), (pnl > 0).mean() * 100


if __name__ == "__main__":
    configs = [
        ("baseline (current)", 2.0, 2.0, 5.0, 0.7, 1.0),
        ("no overlay", 0.0, 0.0, 0.0, 1.0, 1.0),
        ("skew 1", 1.0, 2.0, 5.0, 0.7, 1.0),
        ("skew 3", 3.0, 2.0, 5.0, 0.7, 1.0),
        ("skew 4", 4.0, 2.0, 5.0, 0.7, 1.0),
        ("breach 1/5", 2.0, 1.0, 5.0, 0.7, 1.0),
        ("breach 3/5", 2.0, 3.0, 5.0, 0.7, 1.0),
        ("breach 2/10", 2.0, 2.0, 10.0, 0.7, 1.0),
        ("breach 2/2", 2.0, 2.0, 2.0, 0.7, 1.0),
        ("width 0.5-1.0", 2.0, 2.0, 5.0, 0.5, 1.0),
        ("width 0.8-1.0", 2.0, 2.0, 5.0, 0.8, 1.0),
        ("width 0.7-1.2", 2.0, 2.0, 5.0, 0.7, 1.2),
        ("skew3 breach3/5 w0.6-1.0", 3.0, 3.0, 5.0, 0.6, 1.0),
    ]
    for name, skew, bt, bs, es, ls in configs:
        mean, std, mn, pct = evaluate(skew, bt, bs, es, ls)
        print(f"{name:28s}  mean {mean:+7.2f}  std {std:6.2f}  min {mn:+7.2f}  prof {pct:5.1f}%")
