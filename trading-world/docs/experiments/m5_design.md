# Experiment M5: design notes (written before pre-registration)

**Aim:** stage 6 again (markets that change). Experiment M4's verdict stands: it failed 4 of 5 markets. M5 tests
version 5 on the same markets, with the same references and pre-registered criteria as M4, and fresh seeds.

## Why version 5: M4's own training records

The agents' recorded in-sample likelihoods, on training experience only, showed the regime fit was at fault, not the
choice of the number of regimes.

- **S6X, agents 3–5.** A 2- or 3-regime fit explained their experience no better than one regime (gain about 0 nats).
  For agent 4 the 2-regime fit was 3,359 nats worse. Expectation-maximisation never found the storm regime from
  version 4's starting points. The agents that did find it gained 108–194 nats.
- **S6, agent 2 (a false storm).** The extra "regime" lasted about one bar, i.e. single crashes treated as a mood.

## Changes (fit only; everything else is version 4)

1. **More starting strategies.** Besides version 4's start, the fit also starts from stretches of experience grouped
   by risk alone (volatility and extreme-move frequency), and from stretches grouped by pattern alone. It keeps the
   best training fit, with 2 random starts per strategy, so 6 fits per regime count instead of 4.
2. **A regime lasts at least 50 bars.** The transition matrix's diagonal is floored at 1 − 1/50. Single extreme moves
   belong to the return mixture's components.
3. **Variance floor.** A regime's volatility scale is at least 0.2. The first trial found a singular fit on S6 seed
   802: a regime with volatility scale about 0 collapsed onto the identical synthetic crash returns (exactly −10%),
   giving a spurious likelihood gain (held-out t = 4.2). This is the standard singularity of mixture fitting.

## Trial (real simulator, fresh seeds 801–806, below the ledger's ranges; 20 test runs each)

| Seed | S6X regimes | S6X growth: v5 / true-model filter / v5's own fixed model | S6 regimes |
|---|---|---|---|
| 801 | 2 | +0.75 / +0.72 / −0.06 | 1 |
| 802 | 2 | +1.19 / +1.19 / +0.16 | 1 |
| 803 | 2 | +1.99 / +2.16 / −0.18 | 1 |
| 804 | 2 | +1.65 / +1.73 / −0.32 | 1 |
| 805 | 3 | +1.53 / +1.60 / 0.00 | 1 |
| 806 | 2 | +1.55 / +1.58 / +0.01 | 1 |

The S6 regime counts were taken after the variance floor; the 802 false storm before the floor is recorded above.
For seed 802, S6X learned crash rates of about 1/48 (storm) and 1/286 (calm), against true values of 1/50 and 1/300.

S1X, S2X and S1C are not re-tried here; M5 itself checks them with the same criteria as M4.
