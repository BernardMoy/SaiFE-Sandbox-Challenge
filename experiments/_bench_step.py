"""One-off benchmark: time env.step() at num_trajectories=30 vs 100."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from SAiFE_gym.challenge import ScenarioConfig, create_environment
from SAiFE_gym.agents.BaselineAgents import HoldToken0Agent


def bench(num_trajectories, n_steps=1000, seed=42):
    cfg = ScenarioConfig(num_trajectories=num_trajectories, n_steps=n_steps)
    env = create_environment(cfg, seed=seed)
    agent = HoldToken0Agent(env)
    obs, _ = env.reset()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        action = agent.get_action(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
    dt = time.perf_counter() - t0
    return dt


if __name__ == "__main__":
    for n in (30, 100):
        dt = bench(n, n_steps=1000)
        print(f"num_trajectories={n:4d}  full-episode (1000 steps) time = {dt:.3f}s  ({dt/1000*1000:.3f} ms/step)")
