# Weighted random bot

## Goal

A new random bot that does not bid completely at random: the weight of each call depends on suit lengths,
points, and the bidding of partner and opponents. In the user's words: "Jestem ciekaw jak dobry będzie bot
zbudowany na zaledwie kilku matematycznych decyzjach i losowaniu. Nie ma być on dobry ale może być lepszy niż
losowy bot" (curious how good a bot built on only a few mathematical decisions and randomness can be; it does
not have to be good, but should beat today's random bot, which almost always ends in a 7-level contract).

It is a comparison baseline on the ladder, between `random` and `pass`, plus a small ES experiment: how much
do a handful of hand-picked weights gain when tuned.

Working means:

- `load("weighted")` returns the bot with hand-picked weights (name `weighted-random` in tables);
  `load("weighted:<path.json>")` returns it with weights read from a JSON file (e.g. tuned ones).
- It works everywhere a policy spec works: `bridgebid eval weighted`, `bridgebid ladder`, `bridgebid show`.
- A tuning script finds better weights with a simple ES and writes them to `results/weighted_tuned.json`.

### Decision model ("score + softmax")

Every legal call gets a score that is a few weights × a few features; the call is sampled from
`softmax(score / T)` over legal calls only. Sketch (exact form is the implementer's choice, the ingredients
are not):

```
score(bid L,s) = a·len(s) + b·partner_fit(s) − c·opp_suit(s) − d·max(0, L − target_level(pair points, fit))
score(Pass)    = e·max(0, auction_level − target_level)          (+ base weight)
score(X)       = f·HCP + g·len(opponents' suit) + h·max(0, their level − their estimated level)
score(XX)      = small constant
P(call)        = softmax(score / T) over legal calls
```

- **Points:** HCP (A4 K3 Q2 J1, `cards.hcp`) plus 1 length point per card above 4 in a suit.
- **Partner:** a strain partner bid → assume ~4 cards there (fit bonus); each partner bid adds assumed
  points (e.g. ~12 for an opening, ~8 for a response), a partner pass lowers them. Own points + partner's
  assumed points → estimated pair strength → target level.
- **Opponents:** penalty for bidding a strain the opponents bid; the weight of a double grows with own
  points, length in their suit, and how far they bid above their estimated strength.
- About 8–12 weights in total, including `T`, held in one dataclass with hand-picked defaults.

## Constraints

- Implemented like the rule bot: features computed vectorised in NumPy over the whole batch (reuse
  `rule_bot.auction_features` where it fits), then sampled with `mx.random.categorical(score / T, key=key)`
  using the `key` passed to `act`. No other randomness (no global or NumPy RNG in `act`).
- Same state + same key → same calls, also with batch size 1: duplicate matches and the `--serve` replay
  check in `ui/server.py` ("call N is not what X bids") depend on it.
- Only legal calls: illegal calls get zero probability via the mask.
- All calls natural; no conventions (no Stayman, transfers, strong 2♣, Polish Club).
- Tuning script: simple ES (e.g. (1+λ) or a small CMA-like ES in NumPy) over the weights; fitness = duplicate
  IMPs/board on fixed deals from the **train** split (never eval), averaged over a pool of opponents (default
  brl FSP, brl SL and `rule-bot`; first `rule-bot` alone, changed after that run overfit it); explicit seeds, so a run
  with the same seed gives the same weights. `--minutes` budget; a ~1 minute smoke run. Longer runs only when
  the user says the Mac is free (it is shared with another project).
- Python 3.12, MLX only (no torch/JAX in the bot or the script); `ruff check` clean.

## Out of scope

- Conventions or any artificial calls.
- A deterministic `T=0` (argmax) variant as a separate bot.
- Making the tuned weights the default: `weighted` stays the hand-picked reference; tuned weights load via
  `weighted:<file>`.

## Acceptance criteria

- [ ] `load("weighted")` returns a `Policy` named `weighted-random`; `load("weighted:<file>.json")` loads
      weights from that file.
- [ ] `"weighted"` is in `DEFAULT_LADDER` (first, then `weighted-tuned`; `random` was dropped from the ladder on 20 Nov 2026); `bridgebid ladder`, `eval` and `show`
      include it without other changes.
- [ ] Over 60 calls on 512 eval deals it only makes legal calls.
- [ ] Same state and key → identical calls, for a batch of 512 and for batch size 1.
- [ ] Hand examples: dealer with 0 HCP → Pass is the most likely call; 20 HCP with 6 spades → spades is the
      most likely strain among bids; a strain the opponents bid is never more likely than own long suit of
      the same level.
- [ ] Hand-picked weights beat `random-legal` by ≥ +5 IMPs/board over 10,000 eval boards (duplicate).
- [ ] Self-play diagnostics (`bridgebid eval weighted`): 7-level contracts < 5% of boards, pass-out < 40%.
- [ ] Tuning script smoke run (a few generations, few deals) writes a JSON that `load("weighted:<file>")`
      reads; same seed → same JSON.
- [ ] After a real tuning run (on the user's go), tuned weights beat hand-picked ones head-to-head by more than
      2 standard errors over 10,000 eval boards.
- [ ] README: `weighted` row in the ladder table with its 10,000-board results, a line in the ladder list, the
      tuning command.
- [ ] `uv run ruff check . && uv run pytest` passes; skips (missing dataset) reported, not counted as passes.

## Files affected

- `bridgebid/policies/weighted_random.py` (new): weights dataclass, features, scores, `WeightedRandom`
  policy, JSON load/save of weights.
- `bridgebid/policies/base.py`: `load()` handles `weighted` and `weighted:<path>`; docstring updated.
- `bridgebid/policies/rule_bot.py`: only if `auction_features` needs extra fields (e.g. every strain partner /
  opponents bid, their highest level); existing fields unchanged.
- `bridgebid/eval/ladder.py`: `"weighted"` and `"weighted-tuned"` replace `"random"` in `DEFAULT_LADDER`.
- `scripts/tune_weighted.py` (new): ES tuning against the rule bot on train deals, `--minutes`, `--seed`,
  `--out` (default `results/weighted_tuned.json`).
- `tests/test_weighted_random.py` (new): legality, reproducibility, hand examples, tuning smoke; the 10,000-board
  match vs random and the diagnostics check marked `@pytest.mark.slow`.
- `README.md`: ladder list and table, tuning command.

## Verification

1. `uv run ruff check . && uv run pytest` — new tests pass (dataset-dependent ones skip only if the dataset
   is missing; report it).
2. `uv run pytest -m slow -k weighted` — ≥ +5 IMPs/board vs random, 7-level < 5%, pass-out < 40%.
3. `uv run python scripts/tune_weighted.py --minutes 1 --seed 0 --out runs/weighted_smoke.json` twice →
   identical JSON; `uv run bridgebid eval weighted:runs/weighted_smoke.json --opponents random rule` runs.
4. On the user's go: a longer tuning run, then
   `uv run bridgebid ladder weighted weighted:results/weighted_tuned.json random pass rule --boards 10000`
   → tuned beats hand-picked by > 2 SE; numbers go into the README table.
5. End to end as a user: `uv run bridgebid eval weighted` prints the ladder results and a system card
   with sensible openings by HCP (no 7-level flood), and `uv run bridgebid show brl:sl --opponents weighted`
   opens the viewer with readable `weighted-random` auctions.

## Open questions

- Should the viewer show the bot's top-3 call probabilities (today only neural policies have them)? The user
  neither asked for nor excluded it; default: not in this change.
- Should the bot also serve as an opponent in PPO/ES training (fictitious self-play pool)? Not decided;
  default: no trainer changes.
- Exact feature formulas and default weight values are left to the implementer, within the ingredients above.
