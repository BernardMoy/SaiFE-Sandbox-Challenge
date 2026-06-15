# (Handwritten!) Solution Description

## Background

We want to maximise the profits when trading on Uniswap v3, where we can specify a range to put liquidity in.
A smaller range means liquidity is concentrated, although earning more if trades happen in the range, they are less likely to fall into the range.
We can choose to rebalance the range that allows it to capture new trades, but each rebalancing would cost a `gas_cost` amount.

Price (current_tick) exceeding the upper limit is more harmful than the lower limit:
As the upper limit force all assets sold to become cash, with a capped value.
The lower limit will make you hold all risky asset which may benefit from a later recovery.

## Implemented: Arbitrage & Rule Based Approach

Assumption: Upward exists are mostly caused by arbitrage chasing the price. When the mid price rises above the current price, the current price will also increase to catch it up.

Design principle: Proactively rebalances before the current tick exits the range. The strong assumption above allow our range to be as narrow as possible (1).

### Rebalance occur when:

1. During the first iteration
2. The following 3 conditions are met:

- Current tick is out of range of the 1-tick band width.
- Pool price is within 2 ticks of the midprice. This prevents the arbitrage flow to shoot past the current price.
- Exceeded the 4 steps cooldown between rebalances as we have to pay the gap price.

### Risks:

1. The assumption, while work in the simulation environment provided, may not hold true in real world complex trading environment.
2. The solution is sensitive to hyper-parameters such as width=1, and cooldown=4.

### Future development:

A self-calibrating agent that adjusts the hyper-parameters based on the current market dynamics.

One good place to start is volatility (which is kept constant in this simulation): High volatility favours a wider range, low volatility favours a narrower range.

## (Bernard's part) CMA-ES Evolutionary Algorithm Approach

### States and Actions

This problem can be treated as a reinforcement learning algorithm, where:

- States: (midprice-curprice), distance to lower, distance to upper, time left, gas cost
- Actions: 3 pair tuple (new_lower, new_upper, hold_flag) where flag=-1 means rebalance, 1=hold
- Reward: PnL (Profit and Loss). One example is the simulated profit - baseline profit

### MLP

5 (states) --> tanh --> 8 (Hidden) --> tanh --> 3 (Action)

Target: Optimise the parameterised policy below where θ = [w1, b1, w2, ...] which is a 1\*75 array:

```
action = MLP(state, θ)
```

### CMA-ES Algorithm

This is an evolutionary algorithm (survival of the fittest).

Assumption: Search distribution can be approximated by a multivariate gaussian distribution.

Fitness function: Simulated profit on the training seeds by running the simulated environment.

Steps:

1. Sample 8 θ candidates from the normal distribution N
2. Get the fitness score of them by running simulation
3. Weights are reassigned by giving more weights to the elites
4. Re-fit the distribution N using the new weighted mean

## Additional Improvements

In addition, the following non-ML improvements can be deployed.

1. Skewed range by mid price - cur price. If mid price > cur price, then skew to the right as we expect the price to increase.
2. Gradually increase the width: risk tolerance gradually decrease from the beginning.
3. Rebalance only when exceeded the upper bound for a number of steps, to minimise noise.
4. Cooldown in rebalancing.
5. Less rebalancing towards the end as we are less likely to get the profits that cover the gas cost.

# Running the Custom Agent

Open the venv and do `pip install -r requirements.txt`

Set the seeds in env variable:

```powershell
$env:HACKATHON_SEEDS = "5000,5001,5002"
```

Run the custom agents (This takes around 7 mins for a single seed)

```powershell
python experiments/hackathon_simulation.py --mode practice --submissions-dir implementations
```

## Expected Result

- Arbitrage-based approach (`gap_gated_tight`) rank first with +193 profit
- Evalutionary algorithm based approach (`cma-es`) rank second with +66 profit

  Both beating the baseline by a number of multiples.

```
==============================================================================================
Hackathon leaderboard | mode=practice | seeds=3 | paths/result=300
==============================================================================================
Rank | Agent | Kind | Mean | Std | Median | Min | Max | Profitable

---

1 | gap_gated_tight | submission | +193.43 | 44.78 | +194.68 | +73.65 | +350.21 | 100.0%
2 | cma_es | submission | +65.87 | 23.21 | +65.79 | +10.56 | +138.74 | 100.0%
3 | HoldToken0 | baseline | +24.42 | 22.47 | +21.18 | -26.30 | +102.24 | 88.3%
4 | DeployOnceWide | baseline | +21.47 | 18.95 | +20.45 | -17.16 | +80.40 | 87.7%

---
```

## PnL Histogram

See `experiments/figures/local` where the dotted line represent the mean profit and loss.
