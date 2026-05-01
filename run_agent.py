import argparse
from datapilot import DataPilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataPilot (legacy CLI — prefer cli.py).")
    parser.add_argument("--data",        required=True,      help="Path to CSV dataset")
    parser.add_argument("--target",      required=True,      help="Target column name or 'auto'")
    parser.add_argument("--output_root", default="outputs",  help="Outputs folder")
    parser.add_argument("--seed",        type=int,   default=42,  help="Random seed")
    parser.add_argument("--test_size",   type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--max_replans", type=int,   default=1,   help="Max replans")
    parser.add_argument("--cv",          action="store_true",     help="Enable 5-fold CV")
    parser.add_argument("--quiet",       action="store_true",     help="Reduce logs")
    args = parser.parse_args()

    agent   = DataPilot(verbose=not args.quiet)
    out_dir = agent.run(
        data_path=args.data,
        target=args.target,
        output_root=args.output_root,
        seed=args.seed,
        test_size=args.test_size,
        max_replans=args.max_replans,
        use_cv=args.cv,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
