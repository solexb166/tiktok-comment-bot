# Phase 2 gate: synthetic learning experiments

Criteria were fixed before running (docs/decisions/0002-phase2-gate-criteria.md). Agents are trained on one synthetic market path and tested on a different, unseen path.

| Task | Reward | Seed | Test growth (log) | t | Deaths | Mean gross | Mean net | Result |
|---|---|---|---|---|---|---|---|---|
| S1 | R1 | 0 | +0.6363 | 13.87 | 0% | 2.035x | +0.061x | ✅ growth +0.636 vs null 95th pct -0.078 |
| S1 | R1 | 1 | +0.5674 | 22.82 | 0% | 1.881x | +0.842x | ✅ growth +0.567 vs null 95th pct -0.055 |
| S1 | R1 | 2 | +0.6041 | 20.95 | 0% | 1.863x | -0.167x | ✅ growth +0.604 vs null 95th pct -0.069 |
| S2 | R1 | 0 | +0.1129 | 10.03 | 0% | 0.956x | +0.096x | ✅ agrees with regime 96% of the time, growth +0.113 |
| S2 | R1 | 1 | +0.1742 | 11.33 | 0% | 1.319x | -0.120x | ✅ agrees with regime 98% of the time, growth +0.174 |
| S2 | R1 | 2 | +0.1921 | 11.06 | 0% | 1.612x | +0.127x | ✅ agrees with regime 95% of the time, growth +0.192 |
| S3 | R1 | 0 | -0.0077 | -3.75 | 0% | 0.159x | +0.055x | ✅ mean gross exposure 0.159x |
| S3 | R1 | 1 | -0.0283 | -6.83 | 0% | 0.390x | +0.043x | ❌ mean gross exposure 0.390x |
| S3 | R1 | 2 | -0.0152 | -4.20 | 0% | 0.338x | -0.286x | ❌ mean gross exposure 0.338x |
| S4 | R1 | 0 | -0.0050 | -0.98 | 0% | 0.499x | -0.153x | ✅ growth -0.0050, t = -0.98 |
| S4 | R1 | 1 | +0.0089 | 1.24 | 0% | 0.664x | -0.061x | ✅ growth +0.0089, t = 1.24 |
| S4 | R1 | 2 | +0.0220 | 1.20 | 0% | 1.258x | +1.041x | ✅ growth +0.0220, t = 1.20 |
| S5 | R1 | 0 | +0.0024 | 0.42 | 0% | 0.536x | -0.006x | ✅ growth +0.0024, t = 0.42 |
| S5 | R1 | 1 | -0.0088 | -0.71 | 0% | 0.883x | +0.295x | ✅ growth -0.0088, t = -0.71 |
| S5 | R1 | 2 | -0.0026 | -0.97 | 0% | 0.354x | -0.131x | ✅ growth -0.0026, t = -0.97 |
| S6 | R1 | 0 | -0.8596 | -1.82 | 45% | 4.139x | +4.038x | ❌ gross 4.14x (optimal ~2.9x), deaths 45% |
| S6 | R1 | 1 | +0.1043 | 1.47 | 0% | 0.528x | +0.488x | ❌ gross 0.53x (optimal ~2.9x), deaths 0% |
| S6 | R1 | 2 | -0.6108 | -1.29 | 40% | 5.143x | +5.104x | ❌ gross 5.14x (optimal ~2.9x), deaths 40% |
| S6 | R2 | 0 | -0.1817 | -0.61 | 15% | 1.843x | +1.799x | ❌ gross 1.84x vs R1 4.14x, deaths 15% vs 45% |
| S6 | R2 | 1 | +0.1722 | 0.74 | 5% | 1.809x | +1.751x | ✅ gross 1.81x vs R1 0.53x, deaths 5% vs 0% |
| S6 | R2 | 2 | -1.7113 | -3.72 | 70% | 7.236x | +7.175x | ❌ gross 7.24x vs R1 5.14x, deaths 70% vs 40% |
| S7 | R1 | 0 | -0.0814 | -0.24 | 10% | 2.684x | +2.637x | ✅ net leverage 2.64x (optimal 5x) |
| S7 | R1 | 1 | +0.2486 | 1.25 | 0% | 1.347x | +1.294x | ❌ net leverage 1.29x (optimal 5x) |
| S7 | R1 | 2 | -0.6831 | -1.68 | 30% | 1.960x | +1.826x | ❌ net leverage 1.83x (optimal 5x) |
| S7 | R2 | 0 | +0.1584 | 0.53 | 5% | 1.737x | +1.702x | ✅ net leverage 1.70x, deaths 5% |
| S7 | R2 | 1 | +0.0677 | 0.42 | 0% | 1.242x | +1.233x | ✅ net leverage 1.23x, deaths 0% |
| S7 | R2 | 2 | -0.3316 | -1.07 | 10% | 2.055x | +2.031x | ✅ net leverage 2.03x, deaths 10% |

## Gate summary (majority of seeds must pass)

- ✅ S1 / R1: 3 of 3 seeds pass
- ✅ S2 / R1: 3 of 3 seeds pass
- ❌ S3 / R1: 1 of 3 seeds pass
- ✅ S4 / R1: 3 of 3 seeds pass
- ✅ S5 / R1: 3 of 3 seeds pass
- ❌ S6 / R1: 0 of 3 seeds pass
- ❌ S6 / R2: 1 of 3 seeds pass
- ❌ S7 / R1: 1 of 3 seeds pass
- ✅ S7 / R2: 3 of 3 seeds pass

## Reproducibility

git ad8214f9e3, Python 3.13.18, torch 2.15.0+cpu, SB3 2.9.2, gymnasium 1.4.0, numpy 2.5.3. PPO {'n_steps': 1024, 'batch_size': 512, 'n_epochs': 10, 'learning_rate': 0.0003, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.0}. Policy {'net_arch': [64, 64], 'log_std_init': -0.5}. Env {'decision_interval': 1, 'obs_resolution': 1, 'window': 30}. 20 test episodes x 1000 decisions. Train steps per agent: S1 1,000,000, S2 1,000,000, S3 1,000,000, S4 1,000,000, S5 1,000,000, S6 3,000,000, S7 3,000,000. Data hashes and seeds are in each run's JSON.
