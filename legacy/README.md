# Legacy: genetic-algorithm version (Apr 2025, v0.0.0 – v1.2.8)

The first version of this project, kept for history. It is **not** used by the rewrite (`bridgebid/`).

- `src/` – genetic algorithm over small MLPs for *uncontested* North–South bidding
  (tournament selection, uniform crossover, Gaussian mutation, islands, novelty archive).
- `generator/` – C++ deal generator using Bo Haglund's DDS. For each N/S hand pair it solved 64 random
  E/W layouts and stored the average score of every contract for N and S as declarer.
  `generator/data/` (110 MB of CSV) and `generator/dds/` are local only (gitignored).
- `archive/`, `pure_statistics/` – config + statistics of every run. The saved populations
  (`population.json.gz`, ~590 MB) are local only (gitignored).
- `CHANGELOG.md`, `run_config.yaml`, `change_ideas`, `next_changes` – the experiment log.

The last commit (`v1.2.8`) contains an unfinished numpy → PyTorch refactor that does not run.

## Why it never got past "always pass"

Verified on 19,200 deals of its own data when the project was restarted (Nov 2026):

1. **Strength feature was wrong.** The generator stores card index 0 = Ace, but `batch.py` gave points
   to indices 9–12, i.e. the 5-4-3-2. Correlation with the best achievable score: real HCP +0.70,
   the feature the network saw −0.24. From v1.2.1 the network could not see hand strength.
2. **Bids were not in bridge order.** Data and legality mask were suit-major (1♣…7♣, 1♦…7♦, …), so
   7♣ → 1♦ was legal and 1♥ → 2♣ was not. The viewer decoded bids, suits and ranks differently, so the
   printed auctions were garbled.
3. **Best score used `min()` for South** – wrong on 27% of deals, and the agent was penalised for
   beating the wrong target.
4. Smaller issues: diversity weight imported by value (phase schedule never applied), novelty measured
   between weights instead of behaviour, vulnerability never fed to the network, a reward bonus for long
   auctions, noisy fitness (sampled bids, 128 deals per generation), one Python forward pass per bid.

Mean IMPs lost per deal versus the best contract: always pass 4.56, always 1NT 6.26, a simple rule bot
that sees both hands 2.33, best evolved agents 3.1–5.0.
