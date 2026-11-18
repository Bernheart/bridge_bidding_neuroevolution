# Board viewer

## Goal

"A small UI where I can see how the model is doing": a page that shows, for one board at a time, the four
hands, dealer, vulnerability and every auction the model had on that board, with arrows to go to the next or
previous board. It is for the user (a Polish Club player) to judge the model's bidding in bridge terms, and to
bid a board themself with the model as partner.

Working means:

- `bridgebid show POLICY` plays a range of fixed eval boards and writes one self-contained HTML file, then
  opens it in the browser. View only.
- `bridgebid show POLICY --serve` starts a local server with the same page plus a "Bid this board" mode.
- `bridgebid show --terminal ...` keeps today's terminal output.

### Per board (viewer)

- Board number, dealer, vulnerability, the four hands in a deal diagram, HCP next to each player's name.
- The full double-dummy table of the deal (4 declarers × 5 strains).
- One duplicate pair per opponent: table 1 = model NS vs opponent EW, table 2 = opponent NS vs model EW. With
  the full ladder that is 7 opponents → 14 auctions on the board.
- For each auction: the auction grid (W N E S, starting at the dealer), final contract, declarer, double-dummy
  tricks, NS score; for each pair the model's IMPs on the board.
- For each call made by the model: its top-3 calls with probabilities.

### Navigation and selection (viewer)

- ← / → buttons and the keyboard arrow keys go to the previous / next board.
- First / last buttons go to the first / final board of the current (filtered, sorted) list.
- A "board N of M" counter and a box to type a board number and jump to it.
- The current board number is kept in the URL hash (`#37`), so reloading or sharing the file keeps the place.
- In-page swing controls, no re-run needed:
  - swing = total IMPs over all opponents, or IMPs against one opponent chosen in the page;
  - filter: only boards with |swing| ≥ X;
  - sort: board order, worst swing first, best swing first.

### Bidding mode (`--serve` only)

- From any board, "Bid this board": the user picks a seat; their partner is the model; the other side is an
  opponent picked from the ladder. Seat and opponent are chosen by a click in the page.
- The bots bid their seats until it is the user's turn or the auction ends; the page offers only legal calls.
- After the auction: the user's contract, double-dummy result and NS score, and IMPs against the model's
  auction on the same board (the table where the model sat in the user's partnership's seats against the same
  opponent).
- Display options, each switchable by a click or a setting in the page:
  - only the user's own hand is shown while bidding; the full deal and DD table appear when the auction ends;
  - bot hints: what the model would bid in the user's seat (top-3 with probabilities);
  - bots' reasons live: the model's top-3 probabilities for each call the bots make during the auction.
- "Everything should be changeable by a click or a setting": the viewer's extras (probabilities, DD table,
  HCP) are toggles too.

## Constraints

- Fast to generate: the default run (full ladder, default 200 boards) on the PPO probe checkpoint writes the
  file in under ~30 s on the M2, and the file stays a few MB at most.
- Opponents default to the full ladder (`DEFAULT_LADDER` in `eval/ladder.py`: random, pass, rule and the brl
  bots), filtered by `available()` so missing brl models are skipped; `--opponents` overrides.
- Boards are consecutive boards of the fixed eval split with their standard dealer/vul cycle
  (`--start`, `--boards`, `--split`, as `bridgebid show` has today).
- The model's IMPs per board must be the duplicate IMPs from `eval/duplicate.py` (same boards, same seed);
  bots bid greedily as in evaluation.
- Default output: for a checkpoint under `runs/<run>/`, `runs/<run>/boards.html`; for any other policy
  `runs/viewer/<name>.html`. `--out` overrides. The file opens in the browser after writing.
- The server uses the Python standard library (`http.server`) and binds to localhost only.
- MLX rules from CLAUDE.md: explicit random keys (no global randomness), one `mx.eval` per step, no numpy
  conversion inside the batched play loop beyond what the existing viewer does per step.
- The HTML/JS/CSS are package files of `bridgebid/ui/` and must ship with the package (hatch wheel).

## Out of scope

- Card play; only the auction is shown or bid.
- Changing training, evaluation numbers or the ladder.
- Platforms such as BBO or Cuebids.
- Any server reachable from other machines.

## Acceptance criteria

- [ ] `bridgebid show POLICY` writes a single HTML file at the default path (or `--out`) and opens it.
- [ ] Each board in the page has 4 hands of 13 cards, dealer, vulnerability, HCP and the 4×5 DD table.
- [ ] Each board has 2 auctions per available opponent (14 with the full ladder), each with contract,
      declarer, DD tricks and NS score, and the model's IMPs per opponent.
- [ ] The model's per-board IMPs equal `eval/duplicate.match(model, opponent, ...)` on the same boards and seed.
- [ ] The model's calls show its top-3 calls with probabilities (toggleable).
- [ ] ← / → buttons and arrow keys, first / last, jump-to-board box and `#N` URL hash all work.
- [ ] Swing filter (|swing| ≥ X) and sort (board order / worst / best), with swing = total or one opponent.
- [ ] `bridgebid show --terminal` prints the same output as today's `bridgebid show`.
- [ ] `bridgebid show POLICY --serve` serves the page on localhost with a "Bid this board" mode.
- [ ] In bidding mode the user picks seat and opponent; partner is the model; only legal calls are offered;
      bots bid until the user's turn or the end of the auction.
- [ ] After a user auction: contract, DD result, NS score and IMPs vs the model's auction on that board.
- [ ] Hidden-hands, bot hints and bots' live reasons are each switchable in the page.
- [ ] Default run (full ladder, 200 boards, PPO probe checkpoint) finishes in under ~30 s; file ≤ a few MB.
- [ ] `uv run ruff check . && uv run pytest` passes.

## Files affected

- `bridgebid/ui/__init__.py` (new package)
- `bridgebid/ui/boards.py` (new): play the duplicate pairs, collect hands, DD table, HCP, auctions,
  contracts, scores, IMPs and the model's probabilities into JSON-ready data.
- `bridgebid/ui/server.py` (new): localhost server for `--serve`: page, board data, bidding API.
- `bridgebid/ui/static/viewer.html` (new): the page (viewer + bidding mode).
- `bridgebid/cli/main.py`: `show` gets HTML by default, `--terminal`, `--opponents`, `--out`, `--serve`.
- `bridgebid/cli/viewer.py`: unchanged behaviour behind `--terminal`.
- `tests/test_ui.py` (new); `README.md` (usage).

## Verification

1. `uv run ruff check . && uv run pytest`: includes a test that needs no dataset (a few synthetic deals in a
   temporary DealBank split): board data has every board, 2×k auctions each, 13-card hands, and IMPs equal to
   `match()` on the same boards; a bidding-API test plays one auction to the end with legal calls only.
2. With the dataset: `time uv run bridgebid show runs/20261117-200803_ppo_probe10min/ckpt/policy.safetensors`
   finishes in under ~30 s; `runs/20261117-200803_ppo_probe10min/boards.html` exists and is a few MB at most.
3. `uv run bridgebid show --terminal --ns rule --boards 1` prints the same as `bridgebid show` before the change.
4. End to end in the browser: open the generated file; step with ← / → keys and buttons, first / last, jump to
   board 37, reload and stay on `#37`; set swing = vs rule, filter ≥ 5, sort worst first; check one board's
   hands and auctions against `bridgebid show --terminal` for the same board number.
5. End to end bidding: `uv run bridgebid show <checkpoint> --serve`, open the localhost URL, "Bid this board"
   as South with the model as North against the rule bot; bid to the end; check the result and IMPs appear,
   and that hidden hands, hints and live reasons switch on and off.

## Open questions

- CLI shape: model as a positional `POLICY` (like `bridgebid eval`) while `--terminal` keeps today's
  `--ns/--ew` flags, or one set of flags for both?
- Show top-3 probabilities in the viewer also for neural opponents (brl bots), or only for the model?
- May the page load JS libraries or fonts from a CDN, or must it work offline with inline code only?
- Should the bidding mode allow bidding a fresh random deal, or only boards from the viewer's range?
- Should the user's own auctions be saved anywhere (e.g. a JSONL next to the run) or only shown?
