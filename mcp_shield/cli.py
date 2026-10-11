import argparse
import json
import os
import tempfile
from pathlib import Path

from .core import inspect, load_model, train, unique_fields
from .data import dataset, split
from .evaluation import evaluate
from .offline import compare_models, compare_observations, count_boundaries, inspect_batch, model_info, projection, read_document, receipt, representation_audit, sample_observations, sensitivity, validation_report
from .server import MAX_BODY, make_server
from .offline import replay_models


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


def validate_outputs(sources, outputs):
    sources = [Path(p).resolve() for p in sources if p]
    outputs = [Path(p).resolve() for p in outputs if p]
    for index, output in enumerate(outputs):
        for other in sources + outputs[:index]:
            if output == other or (output.exists() and other.exists() and output.samefile(other)):
                raise ValueError("Output paths must be distinct from inputs, models and other exports.")


def main():
    parser = argparse.ArgumentParser(description="MCP Shield offline synthetic inspection simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train from reproducible synthetic training families only")
    t.add_argument("--output", default="models/baseline.model.json")
    t.add_argument("--seed", type=int, default=17)
    e = sub.add_parser("evaluate", help="Retrain then evaluate on disjoint held-out synthetic families")
    e.add_argument("--output")
    e.add_argument("--seed", type=int, default=17)
    e.add_argument("--predictions", help="Export all synthetic held-out predictions without observation metadata")
    i = sub.add_parser("inspect", help="Inspect JSON metadata; never execute tools")
    i.add_argument("input")
    i.add_argument("--model", default="models/baseline.model.json")
    i.add_argument("--receipt", action="store_true", help="Include unsigned canonical content identities")
    b = sub.add_parser("batch", help="Inspect 1–100 synthetic observations atomically, offline")
    b.add_argument("input", help="JSON array of complete observation objects")
    b.add_argument("--model", default="models/baseline.model.json")
    b.add_argument("--output")
    info = sub.add_parser("model-info", help="Describe local learned parameters, not safety or authenticity")
    info.add_argument("--model", default="models/baseline.model.json")
    comparison = sub.add_parser("compare-models", help="Compare two validated local model artifacts")
    comparison.add_argument("before")
    comparison.add_argument("after")
    scenario = sub.add_parser("compare", help="Compare exactly two synthetic observations, not remediation advice")
    scenario.add_argument("input")
    scenario.add_argument("--model", default="models/baseline.model.json")
    sensitivity_parser = sub.add_parser("sensitivity", help="Simulate boolean signal toggles; never falsify observations")
    sensitivity_parser.add_argument("input")
    sensitivity_parser.add_argument("--model", default="models/baseline.model.json")
    bounds = sub.add_parser("boundaries", help="Simulate count feature and policy boundaries")
    bounds.add_argument("input")
    bounds.add_argument("--model", default="models/baseline.model.json")
    project = sub.add_parser("project", help="Show validated feature representation without needing a model")
    project.add_argument("input")
    validation = sub.add_parser("validate", help="Report all invalid batch items without loading a model")
    validation.add_argument("input")
    audit = sub.add_parser("audit-data", help="Audit synthetic feature collisions and mixed labels")
    audit.add_argument("--seed", type=int, default=17)
    samples = sub.add_parser("samples", help="Generate bounded fictional observation fixtures for batch inspection")
    samples.add_argument("--family", required=True)
    samples.add_argument("--count", type=int, default=10)
    samples.add_argument("--seed", type=int, default=17)
    samples.add_argument("--output")
    replay = sub.add_parser("replay", help="Compare model verdicts on the same bounded synthetic batch")
    replay.add_argument("input")
    replay.add_argument("--before", required=True)
    replay.add_argument("--after", required=True)
    s = sub.add_parser("serve", help="Serve local dashboard and real model inference")
    s.add_argument("--model", default="models/baseline.model.json")
    s.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "serve" and not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535; use 0 to choose an available port.")
    try:
        if args.command == "evaluate":
            validate_outputs([], [args.output, args.predictions])
        elif args.command == "batch":
            validate_outputs([args.input, args.model], [args.output])
        if args.command == "train":
            rows, _ = split(dataset(args.seed))
            result = train(rows)
            target = Path(args.output)
            write_output(target, json.dumps(result, indent=2) + "\n")
            print(f"Trained {len(rows)} synthetic rows -> {target}")
        elif args.command == "evaluate":
            result = evaluate(args.seed, include_predictions=bool(args.predictions))
            if args.predictions:
                rows = result.pop("predictions")
                write_output(args.predictions, json.dumps({"data": result["data"], "seed": args.seed,
                                                           "heldout_sha256": result["heldout_sha256"], "rows": rows}, indent=2) + "\n")
            report = json.dumps(result, indent=2) + "\n"
            if args.output:
                write_output(args.output, report)
            print(report)
        elif args.command == "inspect":
            with Path(args.input).open("rb") as source:
                raw = source.read(MAX_BODY + 1)
            if not 0 < len(raw) <= MAX_BODY:
                raise ValueError(f"Inspection input must be 1 to {MAX_BODY} bytes.")
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields)
            operation = receipt if args.receipt else inspect
            print(json.dumps(operation(value, load_model(args.model)), indent=2))
        elif args.command == "replay":
            print(json.dumps(replay_models(read_document(args.input), load_model(args.before), load_model(args.after)), indent=2))
        elif args.command == "samples":
            report = json.dumps(sample_observations(args.family, args.count, args.seed), indent=2) + "\n"
            if args.output:
                write_output(args.output, report)
            print(report)
        elif args.command == "audit-data":
            print(json.dumps(representation_audit(dataset(args.seed)), indent=2))
        elif args.command == "validate":
            report = validation_report(read_document(args.input))
            print(json.dumps(report, indent=2))
            if report["invalid_rows"]:
                parser.exit(1)
        elif args.command == "project":
            print(json.dumps(projection(read_document(args.input, MAX_BODY)), indent=2))
        elif args.command == "boundaries":
            print(json.dumps(count_boundaries(read_document(args.input, MAX_BODY), load_model(args.model)), indent=2))
        elif args.command == "sensitivity":
            print(json.dumps(sensitivity(read_document(args.input, MAX_BODY), load_model(args.model)), indent=2))
        elif args.command == "compare":
            print(json.dumps(compare_observations(read_document(args.input), load_model(args.model)), indent=2))
        elif args.command == "compare-models":
            print(json.dumps(compare_models(load_model(args.before), load_model(args.after)), indent=2))
        elif args.command == "model-info":
            print(json.dumps(model_info(load_model(args.model)), indent=2))
        elif args.command == "batch":
            report = json.dumps(inspect_batch(read_document(args.input), load_model(args.model)), indent=2) + "\n"
            if args.output:
                write_output(args.output, report)
            print(report)
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
