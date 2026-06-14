"""
The Concentrator — CMA-ES trained policy.

A tiny numpy-only MLP (5 -> 8 -> 3, tanh activations) maps a 5-feature
observation to the [lower_offset, upper_offset, hold_flag] action.

Inputs (per trajectory):
    0: mispricing_norm = (midprice - sqrt_price**2) / midprice
    1: lower_dist       = (current_tick - lp_tick_lower) / tau
    2: upper_dist       = (lp_tick_upper - current_tick) / tau
    3: time_left        = 1 - time / terminal_time
    4: gas_frac         = clip(gas_cost / portfolio_value, 0, 10)

Outputs -> action:
    lower_offset = tau * tanh(out[0])
    upper_offset = tau * tanh(out[1])
    hold_flag    = tanh(out[2]), forced to -1 until lp_ever_deployed (forces
                   the initial deployment regardless of the network output).

Weights were trained with CMA-ES against the official scenario, optimizing
mean final wealth minus HoldToken0Agent's final wealth on matched price paths
(common random numbers for variance reduction).
"""

import numpy as np


class Agent:
    W1 = np.array([
        [1.04284612, 1.53483016, -0.98335326, 1.27855701, -1.86317237, 2.07473887, 1.41323260, 0.66841369],
        [2.88619831, 2.36357922, -1.21112701, 1.22024561, -0.80115612, -1.54074461, -3.52214261, 1.01840152],
        [0.74298122, -0.94262317, -1.38581890, -1.03653021, 2.79946305, -0.13616075, -0.17541032, 2.90349900],
        [1.62848950, -0.20097571, 0.36874418, 0.34276873, -1.74345902, 1.42151246, 3.31623013, 1.11109587],
        [-0.27546138, -0.03523670, 0.86344354, 1.76250661, -1.66141541, 2.40135092, 1.39200512, 2.39216475],
    ])
    b1 = np.array([1.09315239, 2.58789300, -1.72457793, 1.31841259, -0.47010095, 0.53250057, 0.96225347, 0.86738946])

    W2 = np.array([
        [0.68092012, -2.35190575, 2.26558691],
        [1.92559368, -0.34288254, 1.60942597],
        [1.26748929, 1.48739543, 2.39495789],
        [-0.81649930, -1.02439136, -0.17519897],
        [-0.23693361, -0.13725853, -1.28956224],
        [-2.24259734, -0.43026700, -0.76101243],
        [-0.52105891, -2.08654669, 1.08247565],
        [-1.12405894, -0.44469774, 4.12957587],
    ])
    b2 = np.array([-4.05948154, 3.20847288, 0.38357976])

    def __init__(self, config):
        self.tau = float(getattr(config, "tau", 10))
        self.terminal_time = float(getattr(config, "terminal_time", 1.0))

    def get_action(self, state: dict) -> np.ndarray:
        midprice = state["midprice"]
        sqrt_price = state["sqrt_price"]
        current_tick = state["current_tick"].astype(np.float64)
        lp_lower = state["lp_tick_lower"].astype(np.float64)
        lp_upper = state["lp_tick_upper"].astype(np.float64)
        time = state["time"]
        gas = state["gas_cost"]
        portfolio_value = state["portfolio_value"]
        ever_deployed = state["lp_ever_deployed"]

        mispricing_norm = (midprice - sqrt_price ** 2) / midprice
        lower_dist = (current_tick - lp_lower) / self.tau
        upper_dist = (lp_upper - current_tick) / self.tau
        time_left = 1.0 - time / self.terminal_time
        gas_frac = np.clip(gas / np.maximum(portfolio_value, 1e-3), 0.0, 10.0)

        x = np.column_stack([mispricing_norm, lower_dist, upper_dist, time_left, gas_frac])
        h = np.tanh(x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)

        lower = self.tau * out[:, 0]
        upper = self.tau * out[:, 1]
        hold = np.where(ever_deployed, out[:, 2], -1.0)

        return np.column_stack([lower, upper, hold])
