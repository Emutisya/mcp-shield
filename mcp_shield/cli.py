import argparse
import json
import os
import tempfile
from pathlib import Path

from .core import inspect, load_model, train, unique_fields
from .data import dataset, split
from .evaluation import evaluate
from .server import MAX_BODY, make_server


def write_output(path, text):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    if args.command == "serve" and not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535; use 0 to choose an available port.")
    try:
        if args.command == "train":
            rows, _ = split(dataset(args.seed))
            result = train(rows)
            target = Path(args.output)
            write_output(target, json.dumps(result, indent=2) + "\n")
            print(f"Trained {len(rows)} synthetic rows -> {target}")
        elif args.command == "evaluate":
            report = json.dumps(evaluate(args.seed), indent=2) + "\n"
            if args.output:
                write_output(args.output, report)
            print(report)
        elif args.command == "inspect":
            with Path(args.input).open("rb") as source:
                raw = source.read(MAX_BODY + 1)
            if not 0 < len(raw) <= MAX_BODY:
                raise ValueError(f"Inspection input must be 1 to {MAX_BODY} bytes.")
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields)
            print(json.dumps(inspect(value, load_model(args.model)), indent=2))
        else:
            model = load_model(args.model)
            try:
                server = make_server(model, args.port)
            except OSError:
                parser.exit(2, f"Error: Could not start the local simulator on port {args.port}. Choose another --port or use --port 0 for an available port.\n")
            print(f"Offline simulator at http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except RecursionError:
        parser.exit(2, "Error: JSON nesting is too deep. Use the documented inspection and model schemas.\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
