"""bridgebid command line.

  bridgebid data                          download + convert the DD dataset (~420 MB)
  bridgebid baselines                     download + port the 5 brl reference bots (~75 MB)
  bridgebid ladder [--boards N]           duplicate matrix of the baselines
  bridgebid eval POLICY [--boards N]      POLICY vs the ladder (+ diagnostics)
  bridgebid show POLICY [--serve]         HTML board viewer: POLICY vs the ladder, every board
  bridgebid show --terminal [--ns P] [--ew P]   print boards with auctions
  bridgebid train CONFIG [k=v ...] [--hours H]   train (ppo / es) from a YAML config, with overrides
  bridgebid resume RUN_DIR [--hours H]           continue a run from its last checkpoint

POLICY: pass | random | rule | brl:<name> | runs/<run>/ckpt/policy.safetensors
"""

from __future__ import annotations

import argparse
import sys


def _bank(split: str = "eval"):
    from bridgebid.data.sampler import DealBank

    return DealBank(split)


def cmd_data(args) -> None:
    from bridgebid.data.convert import convert
    from bridgebid.data.download import download

    download()
    convert()


def cmd_baselines(args) -> None:
    from bridgebid.models.brl import download, main

    download()
    main()


def cmd_ladder(args) -> None:
    from bridgebid.eval.ladder import DEFAULT_LADDER, format_matrix, ladder_matrix, save

    rows = ladder_matrix(args.policies or DEFAULT_LADDER, _bank(), n_boards=args.boards)
    print(format_matrix(rows))
    if args.out:
        save(rows, args.out)


def cmd_eval(args) -> None:
    from bridgebid.eval.diagnostics import diagnose, format_diagnostics
    from bridgebid.eval.ladder import DEFAULT_LADDER, evaluate
    from bridgebid.policies.base import load

    bank = _bank()
    pol = load(args.policy)
    res = evaluate(pol, bank, args.opponents or DEFAULT_LADDER, n_boards=args.boards)
    for opp, (m, se) in res.items():
        print(f"{pol.name} vs {opp:<28} {m:+.3f} ± {se:.3f} IMPs/board")
    print(format_diagnostics(diagnose(pol, bank, n_boards=min(args.boards, 4000))))


def cmd_show(args) -> None:
    if args.terminal:
        from bridgebid.cli.viewer import show_boards
        from bridgebid.policies.base import load

        ns = load(args.ns or args.policy or "brl:sl")
        ew = ns if args.ew is None else load(args.ew)
        show_boards(ns, ew, _bank(args.split), start=args.start, n=3 if args.boards is None else args.boards)
        return
    if args.policy is None:
        sys.exit("bridgebid show: POLICY is required (or use --terminal)")
    _show_html(args)


def _show_html(args) -> None:
    import time
    import webbrowser
    from pathlib import Path

    from bridgebid.eval.ladder import DEFAULT_LADDER, available
    from bridgebid.policies.base import load
    from bridgebid.ui.boards import build_board_data, default_out, render_html

    t0 = time.perf_counter()
    bank = _bank(args.split)
    model = load(args.policy)
    specs = available(args.opponents or DEFAULT_LADDER)
    opponents = [load(s) for s in specs]
    n = 200 if args.boards is None else args.boards
    data = build_board_data(model, opponents, bank, start=args.start, n=n, seed=args.seed)
    if args.serve:
        from bridgebid.ui.server import make_server

        server = make_server(model, opponents, bank, data, port=args.port, seed=args.seed)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"{n} boards vs {', '.join(o.name for o in opponents)} in {time.perf_counter() - t0:.1f}s")
        print(f"serving {url}  (Ctrl-C to stop)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    out = Path(args.out) if args.out else default_out(args.policy, model.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} boards vs {', '.join(o.name for o in opponents)}) "
          f"in {time.perf_counter() - t0:.1f}s")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


def cmd_train(args) -> None:
    from bridgebid.config import load_config
    from bridgebid.trainers import run_training

    cfg = load_config(args.config, args.overrides)
    if args.smoke:
        cfg = load_config(args.config, args.overrides + ["smoke=true"])
    run_training(cfg, hours=args.hours)


def cmd_resume(args) -> None:
    from bridgebid.trainers import resume_training

    resume_training(args.run_dir, args.overrides, hours=args.hours)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bridgebid", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    sub.add_parser("baselines").set_defaults(fn=cmd_baselines)
    q = sub.add_parser("ladder")
    q.add_argument("policies", nargs="*")
    q.add_argument("--boards", type=int, default=10_000)
    q.add_argument("--out")
    q.set_defaults(fn=cmd_ladder)
    q = sub.add_parser("eval")
    q.add_argument("policy")
    q.add_argument("--opponents", nargs="*")
    q.add_argument("--boards", type=int, default=10_000)
    q.set_defaults(fn=cmd_eval)
    q = sub.add_parser("show", help="HTML board viewer (or --terminal)")
    q.add_argument("policy", nargs="?", help="the model to show (HTML viewer)")
    q.add_argument("--terminal", action="store_true", help="print boards in the terminal (uses --ns / --ew)")
    q.add_argument("--ns", help="terminal: North-South policy (default brl:sl)")
    q.add_argument("--ew", help="terminal: East-West policy (default: same as --ns)")
    q.add_argument("--opponents", nargs="*", help="HTML: opponents (default: the full ladder)")
    q.add_argument("--boards", type=int, help="number of boards (default 200 in HTML, 3 in the terminal)")
    q.add_argument("--start", type=int, default=0)
    q.add_argument("--split", default="eval")
    q.add_argument("--seed", type=int, default=0, help="HTML: duplicate seed (as in evaluation)")
    q.add_argument("--out", help="HTML file (default runs/<run>/boards.html or runs/viewer/<name>.html)")
    q.add_argument("--serve", action="store_true", help="serve on localhost with a 'Bid this board' mode")
    q.add_argument("--port", type=int, default=8765)
    q.add_argument("--no-open", action="store_true", help="do not open the browser")
    q.set_defaults(fn=cmd_show)
    q = sub.add_parser("train")
    q.add_argument("config")
    q.add_argument("overrides", nargs="*")
    q.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    q.add_argument("--hours", type=float, help="stop with a checkpoint after this many hours")
    q.set_defaults(fn=cmd_train)
    q = sub.add_parser("resume")
    q.add_argument("run_dir")
    q.add_argument("overrides", nargs="*")
    q.add_argument("--hours", type=float, help="stop with a checkpoint after this many hours")
    q.set_defaults(fn=cmd_resume)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
