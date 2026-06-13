"""
The Concentrator — submission template.

Copy this file, rename nothing (the grader looks for a top-level class named
``Agent``, or any class whose name ends in ``Agent``), and replace the body of
``get_action`` with your own liquidity-provision strategy.

---------------------------------------------------------------------------
GRADER RULES
---------------------------------------------------------------------------
• Your submission is ONE Python file, delivered as source. No companion files.
• The ONLY third-party dependency you can rely on is numpy.
    - The grader rejects imports other than numpy.
    - You also CANNOT import SAiFE_gym; use the raw string state keys below.
• NO file I/O: open() and obvious NumPy file-loading calls are blocked.
    - To ship a *trained* policy, inline its weights as numpy literals in this
      file (the grader cannot load a torch model or any weights file).
• The whole file must be ≤ 64 KB of source.
• Per-step and per-episode time limits apply — keep `get_action` cheap and
  vectorized (operate on whole arrays, never Python loops over trajectories).

---------------------------------------------------------------------------
THE CONTRACT
---------------------------------------------------------------------------
__init__(self, config):
    `config` is a read-only namespace of scalars describing the episode (NOT
    the live environment — you can't see the RNG/seed or the future). Fields:
        num_trajectories, n_steps, terminal_time, step_size, initial_wealth,
        tau, num_ticks, exponential_value, fee_tier, gas_cost, swap_fee_rate,
        drift, volatility
    Access as `config.tau`, `config.gas_cost`, etc. Use getattr fallbacks to stay
    robust if a field is ever absent.

get_action(self, state) -> np.ndarray of shape (num_trajectories, 3):
    Called once per step. `state` is a dict of numpy arrays (one row per
    trajectory). The useful scalar keys:
        "sqrt_price"        √P of the pool  (so pool price P = sqrt_price ** 2)
        "current_tick"      pool's current integer tick
        "midprice"          external market price (your fair value)
        "time"              elapsed simulation time
        "lp_tick_lower"     your position's lower tick bound (absolute)
        "lp_tick_upper"     your position's upper tick bound (absolute)
        "lp_ever_deployed"  bool — False until you first deploy
        "gas_cost"          cost charged per rebalance
        "portfolio_value"   your current mark-to-market wealth
    Return one action per trajectory: [lower_offset, upper_offset, hold_flag]
        lower_offset, upper_offset : tick bounds RELATIVE to current_tick,
            in [-tau, tau]. Must satisfy lower_offset < upper_offset.
            (The engine rounds them to integers and clips them to the box for you.)
        hold_flag : <= 0  → rebalance into [current_tick+lower, current_tick+upper]
                    >  0  → hold the existing position (no gas this step)

The engine rounds/clips the two offsets, but it reads hold_flag by SIGN only —
so keep it in [-1, 1] yourself.
"""

import numpy as np


class Agent:
    """
    Full-width, lazy-recenter LP strategy with asymmetric exit handling.

    - Deploys a band spanning the full [-tau, +tau] window around the
      current tick: maximum width => maximum time in range => maximum
      passive fee income => minimum forced rebalances.
    - Holds (no gas) whenever the position is still in range.
    - On a downside exit (price fell below the band), keeps holding: the
      position is now 100% token0, so it tracks the external price like a
      HODL position from here on, on top of whatever fees were already
      earned while in range.
    - On an upside exit (price rose above the band), the position is frozen
      as 100% token1 (cash) and stops tracking price entirely. Rebalance
      (recenter the full-width band on the current tick) to escape that
      trap and resume both price tracking and fee accrual -- but only if
      there's enough remaining time to plausibly earn back the gas, and not
      sooner than `cooldown_steps` after the previous rebalance (avoids
      being whipsawed by a run of upside exits during a sustained uptrend).
    """

    def __init__(self, config):
        self.tau = int(getattr(config, "tau", 10))

        terminal_time = float(getattr(config, "terminal_time", 1.0))
        n_steps = int(getattr(config, "n_steps", 1000))
        step_size = float(getattr(config, "step_size", terminal_time / n_steps))
        self.terminal_time = terminal_time

        # Stop chasing upside exits once fewer than this many steps remain --
        # not enough runway left to earn back the gas cost via renewed
        # tracking/fees.
        min_steps_remaining = max(1, int(0.05 * n_steps))
        self.time_left_floor = min_steps_remaining * step_size

        # Minimum number of steps between rebalances, so a sustained run of
        # upside exits during a trend doesn't trigger a rebalance every step.
        self.cooldown_steps = max(1, int(0.005 * n_steps))

        self._cooldown = None  # lazily-sized per-trajectory counters

    def get_action(self, state):
        current_tick = state["current_tick"]
        n = current_tick.shape[0]

        lp_upper = state["lp_tick_upper"]
        ever_deployed = state.get("lp_ever_deployed", np.zeros(n, dtype=bool))
        time = state["time"]

        if self._cooldown is None:
            self._cooldown = np.zeros(n, dtype=np.int64)

        not_deployed = ~ever_deployed
        upside_exit = ever_deployed & (current_tick >= lp_upper)

        time_left = self.terminal_time - time
        can_afford = (time_left > self.time_left_floor) & (self._cooldown >= self.cooldown_steps)

        rebalance = not_deployed | (upside_exit & can_afford)

        self._cooldown = np.where(rebalance, 0, self._cooldown + 1)

        hold_flag = np.where(rebalance, -1.0, 1.0)

        lower = np.full(n, -self.tau, dtype=np.float64)
        upper = np.full(n, self.tau, dtype=np.float64)

        return np.column_stack([lower, upper, hold_flag])
