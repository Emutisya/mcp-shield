import argparse
import json
from pathlib import Path

from .core import inspect, load_model, train
from .data import dataset, split
from .evaluation import evaluate
from .server import make_server


def main():
    parser = argparse.ArgumentParser(description="MCP Shield offline synthetic inspection simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train from reproducible synthetic training families only")
    t.add_argument("--output", default="models/baseline.model.json")
    t.add_argument("--seed", type=int, default=17)
    e = sub.add_parser("evaluate", help="Retrain then evaluate on disjoint held-out synthetic families")
    e.add_argument("--output")
    e.add_argument("--seed", type=int, default=17)
    i = sub.add_parser("inspect", help="Inspect JSON metadata; never execute tools")
    i.add_argument("input")
    i.add_argument("--model", default="models/baseline.model.json")
    s = sub.add_parser("serve", help="Serve local dashboard and real model inference")
    s.add_argument("--model", default="models/baseline.model.json")
    s.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == "train":
            rows, _ = split(dataset(args.seed))
            result = train(rows)
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Trained {len(rows)} synthetic rows -> {target}")
        elif args.command == "evaluate":
            report = json.dumps(evaluate(args.seed), indent=2) + "\n"
            if args.output:
                target = Path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(report, encoding="utf-8")
            print(report)
        elif args.command == "inspect":
            value = json.loads(Path(args.input).read_text(encoding="utf-8"))
            print(json.dumps(inspect(value, load_model(args.model)), indent=2))
        else:
            server = make_server(load_model(args.model), args.port)
            print(f"Offline simulator at http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
