"""bridgebid command line.

  bridgebid data                          download + convert the DD dataset (~420 MB)
  bridgebid baselines                     download + port the 5 brl reference bots (~75 MB)
  bridgebid ladder [--boards N]           duplicate matrix of the baselines
  bridgebid eval POLICY [--boards N]      POLICY vs the ladder (+ diagnostics)
  bridgebid show [--ns P] [--ew P]        print boards with auctions
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
    from bridgebid.cli.viewer import show_boards
    from bridgebid.policies.base import load

    ns = load(args.ns)
    ew = ns if args.ew is None else load(args.ew)
    show_boards(ns, ew, _bank(args.split), start=args.start, n=args.boards)


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
    q = sub.add_parser("show")
    q.add_argument("--ns", default="brl:sl")
    q.add_argument("--ew")
    q.add_argument("--boards", type=int, default=3)
    q.add_argument("--start", type=int, default=0)
    q.add_argument("--split", default="eval")
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
