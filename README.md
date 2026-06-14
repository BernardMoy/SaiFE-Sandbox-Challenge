# Solution Description

## Background

## Implemented: Arbitrage & Rule Based Approach

Assumptions

3 conditions

## (Bernard's part) CMA-ES Evolutionary Algorithm Approach

States, actions, mlp, fitness, ...

## Additional Improvements

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
