# MCP Shield

### Intelligence needs a trust boundary.

**MCP Shield is an explainable inspection engine for the moment an AI agent
is asked to trust a tool.** It turns caller-observed MCP server behavior into
an advisory **allow, review, or deny** decision, combining a learned behavioral
classifier with explicit safety policy and feature-level explanations.

[![CI](https://github.com/Emutisya/mcp-shield/actions/workflows/ci.yml/badge.svg)](https://github.com/Emutisya/mcp-shield/actions/workflows/ci.yml)

## The invention thesis

AI agents extend their reach through tools. That same connection creates a trust
problem: a useful tool description does not establish that the tool will respect
its permissions, protect secrets, or behave consistently after approval.

MCP Shield explores a different contract: **a tool's claimed capability should be
accompanied by inspectable behavioral evidence before an agent relies on it.**
The working core makes that evidence visible instead of collapsing it into an
opaque score.

The global ambition is a trust infrastructure for agent ecosystems: portable
inspection evidence, accountable review, and independently verified controls
that could help organizations adopt agent tooling without giving every tool
unconditional access.

## Experience the working system

| Step | What MCP Shield does today |
| --- | --- |
| Observe | Accepts bounded metadata and eight caller-supplied behavior signals |
| Inspect | Runs a trained Bernoulli naive Bayes model locally |
| Apply policy | Prioritizes explicit gates for credential requests, scope violations, hidden instructions and unapproved network activity |
| Explain | Shows the verdict, triggered policy and signed feature contributions |
| Challenge | Replays synthetic cases and reports both missed cases and false alarms |

The dashboard makes it possible to change an observation and see how the model
and policy respond. Inference runs in Python, not a prerecorded UI simulation.

### Design choices that matter

- **Policy remains accountable.** Explicit gates can override the learned score.
- **Evidence stays inspectable.** Feature contributions and failure cases are exposed.
- **The core works offline.** No external model service, API key or downloaded weights.
- **Inference is not execution.** Inspection never invokes a tool or follows its instructions.

**Current release:** a working local research prototype using synthetic data.
It inspects supplied observations; it is not yet an MCP transport proxy and
does not collect server telemetry, forward responses or enforce its verdicts.
Those integration boundaries are explicit in the roadmap below.

## Quickstart

Clone the standalone project:

```powershell
git clone https://github.com/Emutisya/mcp-shield.git
Set-Location mcp-shield
```

Python 3.10+; no runtime dependencies, keys, APIs, downloaded weights, or LLMs.
From this project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m mcp_shield train
.\.venv\Scripts\python.exe -m mcp_shield evaluate --output reports\baseline.json
.\.venv\Scripts\python.exe -m mcp_shield inspect examples\catalog.json
.\.venv\Scripts\python.exe -m mcp_shield serve
```

Open **http://127.0.0.1:8765** (use this exact host, not `localhost`).
Try public catalog, metadata drift, credential signal, and unknown publisher.
Edit the JSON and run inference. The self-contained HTML has no CDN resources,
but inference requires the Python backend. Add `?clawpilotTheme=dark` or
`?clawpilotTheme=light` to override the system theme. Stop with Ctrl+C.

The package can optionally be installed with `python -m pip install .` to expose
the `mcp-shield` command. Direct `python -m mcp_shield` execution requires no
installation. Models are locally generated in ignored `models\`.
`train` and `evaluate` accept `--seed`; `serve` accepts `--port`.
Missing or incompatible models produce a CLI error: run `train` first.

`python -m mcp_shield batch observations.json --output reports\batch.json`
inspects a JSON array of 1–100 synthetic observations. The entire bounded UTF-8
document is validated before inference; invalid items identify their zero-based
index, and no partial export is written. Results preserve input order and
include advisory decision counts. This never connects to or executes tools.

Evaluation failure scores are display values rounded to 12 decimal places for
portable reports across Python/libm versions. Classification and policy thresholds
always use full precision, so a displayed score can round onto a threshold without
changing its underlying classification.

The evaluation report includes a fixed descriptive ML threshold sweep (0, .25,
.5, .65, .75, 1). Scores equal to a threshold count positive. These held-out
diagnostics do not select thresholds, tune the model, or alter hard gates.

### Run alongside the other projects

Every project defaults to port 8765. To run all four at once, keep MCP Shield on
8765 and use 8766 for Street to School Guardian, 8767 for Fursa, and 8768 for
Maternity Health Copilot. Run each server in its own terminal.

For an automatically selected available port:

```powershell
python -m mcp_shield serve --port 0
```

Open the URL printed by the server. Its health endpoint is that URL followed by
`/health`. Valid ports are 0 through 65535; invalid values are rejected before
the model is loaded. If a chosen port is busy, select another rather than stopping
an unrelated process.

## Input contract

See [`examples/catalog.json`](examples/catalog.json) for a complete safe sample.
The API accepts **exactly** these keys, with no optional fields or coercion:

| Object | Fields |
| --- | --- |
| `server` | `name`, `publisher`: nonempty strings, maximum 120 characters |
| `tool` | `name`: same limit; `description`: nonempty string, maximum 2000 characters |
| `signals` | All eight fields below, required |

| Signal | Type / bound | Caller-observed meaning |
| --- | --- | --- |
| `credential_request` | boolean | Tool sought authentication secrets outside approved workflow |
| `scope_violation` | boolean | Behavior exceeded granted capability or resource scope |
| `hidden_instruction` | boolean | Instruction content was observed hidden in untrusted material |
| `unapproved_network` | boolean | An unapproved network destination was observed |
| `verified_publisher` | boolean | Caller verified publisher identity through a separate trusted process |
| `metadata_changes` | integer 0–20 | Changes since approval, in the caller's chosen observation window |
| `unexpected_calls` | integer 0–100 | Calls outside expected behavior in that same window |
| `response_bytes` | integer 0–1,000,000 | Size of an observed response, not its contents |

Choose and record an observation window in any future integration; this simulator
does not collect events or normalize rates. Unknown observations should not be
invented as false: stop for human review outside this prototype.
Descriptions are inert text. No response payloads or URLs are executed or fetched.
Never submit secrets, real personal data, or sensitive server metadata.

## Policy and explanations

1. **Deny:** any of the four hard boolean gates is true, regardless of ML.
2. **Review:** otherwise ML score >= 0.65, any metadata drift or unexpected call,
   or publisher verification is false.
3. **Allow:** otherwise no observed gate fired. This does **not** establish safety.

Verdicts are advisory outputs only. A future integration must withhold tools and
responses itself, obtain human approval for review, and fail closed on errors.
There is no automatic execution of a verdict or model output.

The score is a **synthetic-trained, uncalibrated model probability**, not a
measured real-world attack probability. Signed log-odds contributions explain
each present or absent feature; they are correlations, not causal evidence.
Name/publisher text never influences the classifier, and publisher verification
is not inferred from a name or domain.

## The intelligence engine

`core.py` implements **Bernoulli naive Bayes from scratch**, not a keyword rule
masquerading as training. For each class and binary feature, it estimates
`P(feature=1 | class) = (positive_count + 1) / (class_count + 2)` with Laplace
smoothing. A smoothed class prior and present/absent feature log likelihoods
produce posterior log odds and a sigmoid score. All parameters are learned from
training rows. The independence assumption is imperfect, especially where policy
signals correlate with text.

Ten fixed, predeclared features: four gates, unverified publisher, drift, call
anomaly (>2 calls), large response (>65,536 bytes), instruction-related words,
and sensitive-data-related words. Conservative policy reviews even one unexpected
call. Vocabulary features are only weak cues: benign descriptions can match,
and malicious descriptions can evade them.

### Reproducible held-out evaluation

```powershell
python -m mcp_shield evaluate --output reports\baseline.json
python -m unittest discover -s tests -v
```

[`reports/baseline.json`](reports/baseline.json) is produced by that exact
evaluation command, with seed 17. The generator creates 1,200 toy cases:
800 from eight training families, 400 from four wholly held-out families.
Templates are disjoint across partitions, no serialized observation is duplicated,
and unique sample tags and names are ignored by features. Shared binary feature
combinations are intentional: they are the representation being learned, not
cross-split copies. Tests verify these properties.

Labels are sampled latent synthetic risk with overlapping behavior and text
distributions. They are **not validated attack labels**. Some risky toy cases
have no visible signals; some benign cases trigger gates. This explicitly
demonstrates both misses and false alarms. The report includes confusion matrices,
precision/recall/accuracy, an always-benign baseline, decision counts, and the
first twelve ML mistakes. ML evaluation uses threshold 0.5, separate from policy's
0.65 review threshold. Policy metrics combine deny/review as nonallow; review is
not itself blocking. No tuning uses held-out labels. No claim of performance on
real MCP servers follows from these numbers.

Measured seed-17 results (400 held-out **synthetic** cases):

| Method | Accuracy | Precision | Recall | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always benign | 0.5400 | undefined | 0.0000 | 0 | 184 |
| ML at 0.5 | 0.8850 | 0.8920 | 0.8533 | 19 | 27 |
| Combined policy, nonallow positive | 0.6650 | 0.5812 | 0.9728 | 129 | 5 |

The conservative policy trades misses for a substantial review/deny burden.
Of 400 cases, it allows 92, reviews 152, and denies 156. These numbers are not
deployment estimates; the synthetic class balance and feature distributions
were invented for an experiment. The report exposes all confusion counts and
46 ML mistakes rather than hiding them behind an accuracy headline.

## Local HTTP API

`GET /` dashboard; `GET /health` liveness; `POST /api/inspect` inference.
POST uses `Content-Type: application/json`, a Content-Length, and maximum
16,384 bytes. JSON syntax/schema errors return 400 with an actionable message;
oversized bodies 413; unsupported content types 415; unknown routes 404.
The service binds loopback only, checks Host and Origin, has no CORS permission,
and sends a restrictive content-security policy. Inputs are not logged or stored.

It is a single-user local demo, **not an authenticated network service**. Other
local processes can access it, and it has no rate limiter or concurrency quota.
Do not expose it using tunnels or reverse proxies. Model files are plain JSON,
not pickle; only load models you trust.

## Threat model and limitations

Intended future threats include deceptive tool metadata, out-of-scope behavior,
credential seeking, hidden instructions, and unapproved network activity.
The prototype only interprets supplied indicators of those threats. It cannot
observe them itself. A dishonest server can hide behavior or report false signals;
compromised collectors or a local attacker can invalidate every assumption.
Benign tools can trigger false positives. Obfuscated language, multilingual
instructions, multi-step attacks, poisoning, and unknown behaviors may pass.
No raw response scanning, sandboxing, cryptographic verification, MCP transport,
or agent integration exists here. It cannot detect all attacks.

Keep it offline with synthetic inputs. Independent review, representative
consented data, calibration, and rigorous integration work are prerequisites for
even a limited production trial.

## From working core to global trust infrastructure

These are aspirations, not delivered capabilities:

1. **Instrument the boundary:** read-only MCP transport adapters and capability
   manifests; signed publisher provenance, schema drift histories, and trusted
   collectors. Replay recorded, consented observations before enforcement.
2. **Make evidence portable:** open, privacy-preserving safety benchmarks with
   provenance; split by publisher, attack family, and time; measure false alarms
   per operational workload, uncertainty, calibration, and multilingual failure.
3. **Protect without centralizing secrets:** local isolation, least-privilege tool
   brokers, human approvals, independently audited fail-closed controls, and
   privacy-preserving cross-organization incident exchange.
4. **Build an ecosystem safety commons:** interoperable evidence receipts, public
   red-team challenge sets, accessible review interfaces, and community governance
   for agent tooling used in schools, public services, and critical industries.

Go/no-go gates: verify collector integrity; demonstrate held-out real-world
benefit over deterministic-only controls; quantify acceptable workload-specific
false alarms; commission an external assessment; then conduct a bounded pilot.
No dates or fabricated adoption metrics.

## Project map

`mcp_shield\`: schema, model, generator, evaluation, CLI, HTTP server, single-file UI.
`examples\`: safe input. `reports\`: generated baseline. `tests\`: gates, validation,
split integrity, real HTTP inference and errors. `.github\workflows\ci.yml`:
stdlib tests and CLI smoke tests on Python 3.10/3.12/3.13.

Contributions should use safe fictional fixtures, document
assumptions and failures, and never add live exploit payloads or credentials.
