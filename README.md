# bridgebid — learning contract-bridge bidding on a Mac GPU

An AI that learns **competitive contract-bridge bidding** (all four players, auction only) from scratch, scored
by double-dummy results, trained on an Apple-silicon GPU with [MLX](https://github.com/ml-explore/mlx).
Two learners are compared head-to-head on the same environment and the same evaluation:

* **PPO self-play** (reinforcement learning, the main bet), and
* **OpenAI-style evolution strategies** (the neuroevolution side, from the project's roots).

The first version of this project (2025, a genetic algorithm) lives in [`legacy/`](legacy/README.md) together with
an analysis of why it never got past "always pass" (three data/encoding bugs, not tuning).

## Status (Nov 2026)

| Milestone | State |
|---|---|
| M0 restructure, tooling | done |
| M1 data, environment, tests (incl. Pgx/endplay oracle) | done |
| M2 duplicate evaluation, baselines, ported reference bots, viewer | done |
| M3 PPO trainer | done; 10-min probe ≈ rule-bot level; overnight runs next |
| M4 ES trainer | done; 10-min probe +1.2 vs pass; overnight runs next |
| M5 ES vs PPO report · M6 play UI · M7 imitation (Polish Club) & system design | next |

## How it works

**Environment.** A batched auction environment written in MLX (`bridgebid/env/core.py`): thousands of deals
stepped in lockstep on the GPU, ~5M calls/s on an M2. It re-implements the rules, observation (480 bits) and
scoring of Pgx's `bridge_bidding` and is checked against it call by call: legal moves, observations, seat to
act, end of auction and final score agree on 100,000 random games biased toward doubles, redoubles and
pass-outs. Scoring is checked against Pgx and endplay on every contract × doubling × vulnerability × tricks.

**Data.** 12.5M deals with full double-dummy tables from Pgx's public dataset (HF `sotetsuk/dds_dataset`,
Apache-2.0), kept packed on the GPU (32 bytes/deal) and decoded on demand. 300 tables re-solved with endplay
match exactly. Fixed evaluation boards (100K) use the standard 16-board dealer/vulnerability cycle.

**Evaluation.** Duplicate IMPs: every board is played twice with the sides swapped, so the luck of the cards
cancels. Reported as mean ± standard error. The ladder:

| opponent | what it is |
|---|---|
| random | uniformly random legal calls |
| weighted | a weighted random bot (`policies/weighted_random.py`): each legal call gets a score from ~12 weights × a few features (suit lengths, points, a target level from partner's and the opponents' bids), then softmax sampling; natural calls only. `weighted:<file.json>` loads other (e.g. tuned) weights |
| pass | always passes |
| rule | a hand-written natural system (`policies/rule_bot.py`): 15–17 1NT, 5-card majors, weak twos, raises, overcalls, takeout doubles |
| brl:* | the five published models of Kita et al. 2024 (Pgx + PPO), ported to MLX; their scores vs WBridge5 are published, which puts our numbers on a real scale |

Diagnostics catch collapse early (pass-out rate, share of the most common opening, distinct auctions) and
print a "system card": what the dealer opens with each HCP range.

**PPO** (`trainers/ppo.py`). One 4×512 policy plays all four seats. Reward = final score from the acting
side's view. The critic is *privileged*: during training it sees all four hands and the DD table (the policy
never does), which removes most of the variance. GAE with stride 4 (a player's next decision is four calls
later). Curriculum: first one side per deal is forced to pass (uncontested bidding), then full competition plus
fictitious self-play against past snapshots.

**ES** (`trainers/es.py`). A 480-256-256-38 policy (199K parameters). Each generation 512 antithetic members
play the *same* boards in duplicate against a shared opponent (always-pass during the warm-up; then the current
centre, past centres and the rule bot). Fitness = IMPs/board, centred ranks, Adam. A whole chunk of members is
evaluated with one batched matmul per layer. Kill criterion: after the warm-up the centre must beat always-pass
by ≥ 1 IMP/board, otherwise the run stops and reports it — there is no published ES success at bridge bidding,
so this side of the comparison is an honest experiment.

## Results so far

Measured on an Apple M2 (8-core CPU, 10-core GPU, 24 GB), MLX 0.32.4, 17 Nov 2026.

**Ladder** — duplicate IMPs/board for the row policy against the column policy, 10,000 fixed evaluation boards,
± one standard error ([`results/ladder_2026-11-17.json`](results/ladder_2026-11-17.json); full 10×10 matrix via
`bridgebid ladder`). The ported brl models reproduce their published order. The weighted random bots were measured
20 Nov 2026 ([`results/ladder_weighted_2026-11-20.json`](results/ladder_weighted_2026-11-20.json)). With no
conventions, the hand-picked weights beat uniformly random calls by +18.40 ± 0.05 and edge always-pass. Tuning
against the rule bot alone ([`results/weighted_tuned_v1_rule.json`](results/weighted_tuned_v1_rule.json)) only
beat the rule bot; head-to-head it tied the hand-picked weights (+0.04 ± 0.07). Tuning against a pool (brl FSP,
brl SL, rule bot; [`results/weighted_tuned.json`](results/weighted_tuned.json), 370 generations) beats the
hand-picked weights by +0.87 ± 0.07, the first tuned set by +0.65 ± 0.06, random by +22.80 ± 0.03, and improves
against every opponent: twelve tuned numbers and a dice roll come within 1.8 IMPs/board of brl's PPO-from-scratch
bot (5.2B training steps).

| policy | vs pass | vs rule bot | vs brl scratch | vs brl SL | vs brl FSP | published vs WBridge5 |
|---|---:|---:|---:|---:|---:|---:|
| random legal calls | -13.51 ± 0.04 | -13.73 ± 0.05 | -22.73 ± 0.04 | -18.18 ± 0.05 | -20.35 ± 0.05 |  |
| weighted random (hand-picked weights) | +0.28 ± 0.06 | -0.66 ± 0.07 | -3.08 ± 0.08 | -3.19 ± 0.07 | -4.81 ± 0.07 |  |
| weighted random (ES-tuned vs rule bot only) | +0.66 ± 0.06 | +0.70 ± 0.07 | -2.42 ± 0.07 | -2.72 ± 0.07 | -4.08 ± 0.07 |  |
| **weighted random (ES-tuned vs FSP + SL + rule, 42 min)** | +1.64 ± 0.05 | +0.55 ± 0.07 | -1.79 ± 0.07 | -2.28 ± 0.07 | -3.56 ± 0.07 |  |
| always pass | — | -3.25 ± 0.05 | -2.50 ± 0.07 | -4.02 ± 0.06 | -3.93 ± 0.06 |  |
| rule bot (hand-written) | +3.25 ± 0.05 | — | -1.88 ± 0.07 | -2.92 ± 0.06 | -3.65 ± 0.07 |  |
| **ES, 10 min** (ours) | +1.20 ± 0.03 | -1.96 ± 0.05 | -2.63 ± 0.07 | -3.60 ± 0.06 | -4.14 ± 0.06 |  |
| **PPO, 10 min** (ours) | +3.48 ± 0.05 | +0.13 ± 0.05 | -1.52 ± 0.07 | -2.66 ± 0.07 | -3.38 ± 0.07 |  |
| brl PPO from scratch (5.2B steps) | +2.50 ± 0.07 | +1.88 ± 0.07 | — | -0.21 ± 0.07 | -1.54 ± 0.07 | −0.64 |
| brl supervised (WBridge5 imitation) | +4.02 ± 0.06 | +2.92 ± 0.06 | +0.21 ± 0.07 | — | -1.35 ± 0.06 | −0.56 |
| brl SL + PPO | +4.74 ± 0.06 | +3.88 ± 0.06 | +1.75 ± 0.07 | +1.13 ± 0.06 | -0.04 ± 0.06 | +0.88 |
| brl SL + PPO + FSP | +3.93 ± 0.06 | +3.65 ± 0.07 | +1.54 ± 0.07 | +1.35 ± 0.06 | — | +1.24 |

**First 10-minute runs** (not tuned; full overnight runs are next):

* **PPO** (32M calls, still in the uncontested warm-up) already beats always-pass by +3.48 and edges the rule bot
  (+0.13 ± 0.05) in full competitive play. Its self-invented system: pass up to ~11 HCP, 1♥/1♠ as the
  normal openings with 11–15, **1♣ as the strong opening (83% of 18–19 HCP hands)**, 3NT with 20+ — a
  strong-club structure, found from scratch.
* **ES** (180 generations × 524K auctions) spent ~40 generations on the "pass everything" plateau, then broke out
  to +1.20 vs always-pass (kill criterion 1.0 passed) but is still 2 IMPs/board behind the rule bot. Its
  favourite opening is a catch-all 2♠ (55% of openings).

**Throughput (M2 GPU)**

| workload | rate |
|---|---|
| environment (mask + observation + sampling + step) | 4–5.7 M calls/s |
| 4×512 policy forward / fwd+bwd+Adam | 0.8 M / 0.34 M samples/s |
| 4×1024 policy forward / fwd+bwd+Adam | 0.29 M / 0.10 M samples/s |
| ES population forward (480-256-256-38) | 4 M (fp32) – 5 M (fp16) rows/s; CPU 0.5–1.3 M |
| **PPO end to end** (4×512 actor, privileged critic) | **~45–60K calls/s → 1B calls ≈ 5–6 h** |
| **ES end to end** (P=512, 512 boards × 2 tables) | 3–10 s/generation (~55–180K auctions/s) |
| duplicate evaluation, 4×1024 nets | 10,000 boards in ~5 s |


## Quick start

```bash
uv sync                                   # Python 3.12 env with MLX
uv run bridgebid data                     # download + check the DD dataset (~420 MB)
uv run bridgebid baselines                # download + port the reference bots (~75 MB)
uv run bridgebid ladder --boards 10000    # baseline matrix
uv run bridgebid show --terminal --ns brl:pretrained-rl-with-fsp --ew rule --boards 3
```

**Board viewer.** `bridgebid show POLICY` plays 200 fixed eval boards in duplicate against the whole ladder
(the same IMPs as `bridgebid eval`) and writes one self-contained HTML page (a few seconds, under 1 MB), then
opens it: the deal with HCP, the double-dummy table, the 14 auctions of every board with contract, DD result,
score and IMPs, and the model's top-3 calls with probabilities. ← / → step through the boards; boards can be
filtered by swing (total or against one opponent) and sorted worst/best first; `#37` in the URL keeps the place.

```bash
uv run bridgebid show runs/<run_dir>/ckpt/policy.safetensors          # -> runs/<run_dir>/boards.html
uv run bridgebid show brl:sl --boards 500 --start 1000 --opponents rule brl:pretrained-rl --out sl.html
uv run bridgebid show runs/<run_dir>/ckpt/policy.safetensors --serve  # + "Bid this board" on localhost
```

With `--serve` the page also lets you bid any board yourself with the model as partner against an opponent
from the ladder: only legal calls are offered, the result is scored double-dummy and compared in IMPs with the
model's own auction at that table. Hidden hands, the model's hint for your seat and the bots' probabilities
are switches in the page. The server listens on 127.0.0.1 only.

Training (every run writes `runs/<date>_<trainer>_<name>/` with config, metrics, TensorBoard, checkpoints):

```bash
uv run bridgebid train configs/ppo_base.yaml --smoke           # 1-minute end-to-end check
uv run bridgebid train configs/ppo_base.yaml name=night1 --hours 8
uv run bridgebid resume runs/<run_dir> --hours 8               # continues bit-exactly
uv run tensorboard --logdir runs
uv run bridgebid eval runs/<run_dir>/ckpt/policy.safetensors   # ladder + system card
```

Ctrl-C saves a checkpoint, so the Mac can be handed back at any time. Config values can be overridden on the
command line, e.g. `ppo.lr=3e-5 ppo.critic=shared`.

The weighted random bot's weights are tuned by a small (1+λ)-ES against the rule bot on train deals
(`--minutes` is turned into a fixed generation count, so the same seed gives the same JSON):

```bash
uv run python scripts/tune_weighted.py --minutes 1 --seed 0 --out runs/weighted_smoke.json   # smoke
uv run python scripts/tune_weighted.py --minutes 60 --seed 0           # -> results/weighted_tuned.json
uv run bridgebid ladder weighted weighted:results/weighted_tuned.json random pass rule --boards 10000
```

## Tests

```bash
uv run pytest                                   # unit tests + 1,000 golden Pgx games (no JAX needed)
uv run pytest -m slow                           # bit-exact resume
uv sync --group oracle && ORACLE_GAMES=100000 uv run --group oracle pytest -m oracle   # live Pgx + endplay
```

Each bug of the first version has a test: honours count as HCP (not 5-4-3-2), bids are in bridge order
(1NT < 2♣), the best contract considers both declarers, vulnerability is in the observation, and the config is
frozen and saved per run.

## Layout

```
bridgebid/  constants, cards, scoring, config
  data/       download, convert (checks), sampler (GPU decode)
  env/        reference.py (readable spec), core.py (batched MLX), pgx_oracle.py (tests only)
  models/     actor_critic.py, io.py, brl.py (ported baselines)
  policies/   base (pass/random/neural), rule_bot, weighted_random
  eval/       duplicate, ladder, diagnostics
  trainers/   ppo, es, common
  tracking/   run directories, JSONL, TensorBoard
  cli/        bridgebid command, terminal viewer
  ui/         HTML board viewer, localhost bidding server
configs/  scripts/ (bench_mlx, make_fixtures, port_brl, tune_weighted)  tests/  legacy/
```

## Roadmap

* **M3/M4** long PPO and ES runs (overnight on the M2), with ablations: curriculum on/off, privileged vs shared
  critic, network size.
* **M5** ES vs PPO at equal wall-clock: learning curves, ladder table, 100K-board head-to-head, write-up.
* **M6** board viewer and bidding with the model as partner (`bridgebid show`, done); next: fresh random deals,
  saving your own auctions.
* **M7** imitation then RL: OpenSpiel's WBridge5 SAYC data, and **Polish Club (WJ)** from BEN's robot-bid data
  (GPL; data only). Then system design: read the learned system card, pin chosen openings, "update our system".

## Credits

Rules/observation spec and DD dataset: [Pgx](https://github.com/sotetsuk/pgx) (Apache-2.0). Reference bots:
[brl](https://github.com/harukaki/brl), Kita et al. 2024 (Apache-2.0). Scoring/DD cross-checks:
[endplay](https://github.com/dominicprice/endplay) (MIT).
