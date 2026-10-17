# ZN Roll Liquidity: When Should You Roll a 10-Year Treasury Futures Position?
 
This repository contains our team's final project: an analysis of how trading liquidity migrates from the expiring 10-Year U.S. Treasury Note futures contract (**ZN**) to the next contract during each quarterly roll, whether that migration can be predicted, and whether predicting it lowers the cost of rolling a position.
 
All market data comes from **Databento's CME Globex MDP 3.0 dataset (`GLBX.MDP3`)**.
 
> **Status:** Data preparation and prediction notebooks (01–07) are implemented. Execution backtesting remains in progress. Downloaded data and numerical results are not committed.

---


## Team Roles

- **Tech Leader:** Jacob Lerner (created initial README, skeleton of files and project/README outline, and technical workflow)
- **Communication Leader:** TBD
- **Design Leader(s):** Zheng Guo, Christo Karahalios (Both updated project roadmap, design outline and issues)

---
## Project Overview

### Motivation
 
Treasury futures are physically delivered. A trader who is long a contract on **First Position Day** can be assigned delivery, so nearly every position holder moves ("rolls") into the next quarterly contract beforehand. Each March, June, September, and December, this creates a short window in which volume, open interest, and quoted liquidity shift from the front contract to the deferred contract, mostly through the calendar spread market.
 
The timing of that shift matters to anyone holding a futures position:
 
- **Roll too early,** and you trade the deferred contract or the spread while it is still thin.
- **Roll too late,** and you trade the front contract after liquidity has left, while also approaching delivery.


### Research Questions
 
1. **Describe:** For every ZN quarterly roll since 2010, how and when do volume, open interest, calendar-spread activity, and bid–ask liquidity move from the front contract to the next contract?
2. **Predict:** Using only information available on a given day, can we predict when the liquidity crossover will happen, or how far it has progressed?
3. **Execute:** Does timing a roll with our predictions reduce transaction costs compared with simple rules, such as rolling a fixed number of days before First Notice Day or rolling at the open interest crossover?

### Scope
 
We deliberately study **one product: ZN**. CME products differ in calendars, delivery rules, and market structure, and handling those differences would take time away from the analysis itself. If the ZN analysis is complete, we may compare against the Ultra 10-Year (TN) or other Treasury tenors as a stretch goal.

---

## Key Concepts
 
| Term | Meaning in this project |
|---|---|
| **Front contract** | The expiring quarterly ZN contract (e.g., ZNZ6). |
| **Next contract** | The following quarterly contract (e.g., ZNH7). |
| **Calendar spread** | A CME-listed instrument that trades the front and next contracts together (e.g., `ZNZ6-ZNH7`). Most roll volume trades here. |
| **First Notice Day (FND)** | The last business day of the month before the contract month. This is our main time anchor: every roll is measured in **business days relative to FND**. |
| **First Position Day** | Two business days before the first business day of the contract month. Longs holding past this point may be assigned delivery. |
| **Crossover** | The first day on which the next contract's share of combined volume (or open interest) exceeds 50%. |
| **Roll cost** | The estimated cost of moving N contracts from front to next at a given time, either through the spread book or by trading each leg separately. |
 
---

## Technical Workflow

The project is organized as a reproducible pipeline:

1. **Download data** — Retrieve ZN futures, calendar spread, and market statistics data from Databento.

2. **Process contracts** — Identify each quarterly front/next contract pair and align observations relative to First Notice Day.

3. **Build roll panel** — Combine volume, open interest, spread activity, and liquidity measures into a standardized dataset for each roll.

4. **Analyze liquidity migration** — Generate summary statistics and visualizations showing how liquidity moves between contracts.

5. **Model roll timing** — Fit and evaluate prediction models using roll-level out-of-sample validation.

6. **Evaluate execution** — Compare estimated transaction costs from model-based timing against benchmark roll rules.

Where possible, reusable data processing and modeling functions will be kept in `src/`, while `notebooks/` will primarily be used for exploration, visualization, and presentation of results.

---

## Planned Approach
 
### 1. Data Collection
 
All data is pulled from Databento's `GLBX.MDP3` dataset using the `databento` Python client.
 
| Databento schema | Used for | History |
|---|---|---|
| `definition` | Instrument IDs, expirations, spread legs, tick sizes | Full |
| `ohlcv-1d` | Daily volume and prices for outrights and spreads | Full (2010 onward) |
| `ohlcv-1h` / `ohlcv-1m` | Intraday volume profile inside roll windows | Full, roll windows only |
| `statistics` | Daily open interest and settlement prices | Full |
| `bbo-1m` / `tbbo` / `mbp-1` | Top-of-book bid–ask and size during roll windows | As far back as our license allows |
| `mbp-10` | Full depth for a single-roll case study | Most recent one to two months only (license limit) |
 
Symbols are requested by parent (`ZN.FUT`), which returns both outright contracts and calendar spreads. Databento's continuous contracts `ZN.v.0` (volume-ranked) and `ZN.n.0` (open interest-ranked) serve as a reference roll schedule.
 
Raw data is cached locally under `data/raw/` and is **not committed** to the repository.
 
### 2. Roll Panel
 
For each quarterly roll, we build a daily panel indexed by business days relative to FND (about −20 to +5). It includes:
 
- Front and next contract volume and open interest
- Next contract share of total volume and of total open interest
- Calendar spread price, volume, and bid–ask spread
- Bid–ask spread and top-of-book size for each outright
- Business days to FND and to last trading day
### 3. Exploratory Analysis and Visualization
 
- All rolls overlaid on one chart: next contract share vs. days to FND
- Crossover timing by year, compared against Databento's continuous-contract roll dates
- Intraday volume and spread activity around the crossover
### 4. Prediction Models

#### Prediction Target

The primary prediction target is the timing of the volume-based liquidity crossover. For each trading day \(t\), define the next-contract volume share as

$$
S_t = \frac{V^{next}_t}{V^{front}_t + V^{next}_t}
$$

The realized crossover date is defined as the first trading day on which the next contract accounts for more than 50% of combined front- and next-contract volume.

For predictive modeling, the problem is formulated dynamically. On each trading day before the realized crossover, the model estimates the probability that the crossover will occur within the next \(k\) business days using only information available as of that date.

Open-interest crossover is treated separately as a secondary liquidity signal and benchmark rather than being combined with the primary volume-based target.

#### Models
 
Simple baselines come first, because the calendar alone likely explains most of the roll timing:
 
- **Baselines:** roll a fixed N business days before FND; roll at the open interest crossover; roll on Databento's `ZN.v.0` switch date
- **Bayesian hierarchical logistic curve:** each roll's volume-share path is modeled as an S-curve, with a midpoint and steepness drawn from a shared distribution across rolls
- **Discrete-time hazard / survival model:** the probability that the crossover happens today, given it has not happened yet
- **Gradient-boosted trees with quantile loss:** next-day liquidity and bad-case roll cost
Every model is validated **by roll** (leave-one-roll-out or walk-forward), never by randomly shuffling days, so that information from a single roll cannot appear in both training and test data.
 
### 5. Execution Backtest and Fat-Tail Evaluation
 
We compare model-timed rolls against the baselines by the **distribution of roll costs**, not by how accurately each method predicts the crossover date. Because the sample is about 65 rolls and trading costs can be heavy-tailed, the evaluation draws on N. N. Taleb's *Statistical Consequences of Fat Tails*:
 
- Tail diagnostics on roll-window data: kurtosis under aggregation, maximum-to-sum (MS) plots, and the kappa metric
- Mean absolute deviation and upper quantiles of cost reported alongside averages
- An estimate of how many rolls are needed before a difference in average cost can be trusted
- A check of whether a handful of stressed rolls drive any measured improvement
---
 
## Expected Outputs
 
- A reproducible pipeline that downloads ZN data from Databento and builds the roll panel
- Figures describing liquidity migration across every roll from 2010 to 2026
- Fitted prediction models with roll-level out-of-sample results
- A roll-cost comparison table: model-timed vs. baseline roll rules
- A short written summary of findings in `results/`
---


## Repository Structure

```text
├── data/          # Raw and processed data
├── notebooks/     # Exploratory analysis and development
├── src/           # Project source code
├── results/       # Figures, tables, and other outputs
├── README.md      # Project description and instructions
└── requirements.txt
```

## Running the Project

For the corrected crossover timing comparison from Issue #20, run
`python -m src.timing_model` after generating the roll panel with notebook 02.
Alternatively, open `notebooks/09_improve_crossover_timing.ipynb` in Jupyter
and run all cells to view the comparisons directly in the notebook.
This uses existing data without downloading anything and writes comparisons,
a date audit, and a figure under `results/issue-20/`. The historical-median
baseline remains preferred on the current sample. See
[model comparison and limitations](docs/issue-20-model-improvements.md).

Use Python 3.10+ and run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-workflow.txt
export DATABENTO_API_KEY=...  # Windows PowerShell: $env:DATABENTO_API_KEY="..."
python scripts/run_pipeline.py --dry-run
python scripts/run_pipeline.py
```

The full run executes notebooks 01 through 07 in order, always using
`notebooks/` as the working directory. Notebook 01 downloads historical
Databento data and may incur charges; check dataset access and estimated costs
before running it. Data is cached in the git-ignored `data/raw/` and
`data/processed/` directories.

To reuse downloaded raw data without calling Databento again:

```bash
python scripts/run_pipeline.py --start-at 2
```

To run a smaller section, use `--start-at N --stop-at M` (1 through 7).
Later stages require the previous stages' Parquet outputs. The existing data
notebook covers 2020–2026, not the longer 2010–2026 span proposed above.
The workflow has not yet been validated against a full downloaded dataset.

## Project Status

**Current stage:** Notebooks 01–07 implemented. Predictive evaluation is ready for data validation; execution-cost analysis remains open.
