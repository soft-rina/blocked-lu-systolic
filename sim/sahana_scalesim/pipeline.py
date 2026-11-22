#!/usr/bin/env python3
"""Run the CSEN 318 blocked-LU array experiments."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from cycle_model import os_cyc, ws_cyc

KEYS = ("M", "N", "K", "array_h", "array_w")
CALIBRATION = (64, 1024, 1984, 256, 384, 512, 768, 1280, 1536, 1792)
# (matrix size, trailing dimension, block width)
HOLDOUT = ((2048, 960, 64), (2048, 448, 32), (2048, 1344, 32),
           (2048, 1600, 64), (4096, 4000, 32), (4096, 3904, 64))
PAIR_SIZES = (128, 1024, 1984)


def make_case(s, k=64, array=128, shard=False, role="calibration", n=2048,
              dataflow="ws"):
    return dict(
        M=s, N=s // 4 if shard else s, K=k, array_h=array, array_w=array,
        matrix_n=n, block_size=k, trailing_size=s, layout="four" if shard else "single",
        role=role, dataflow=dataflow,
    )


def run_id(case):
    return case["dataflow"] + "_" + "_".join(f"{key}{case[key]}" for key in KEYS)


def planned_cases(group):
    calibration = [make_case(s) for s in CALIBRATION]
    holdout = [
        make_case(s, k=k, array=h, shard=h == 64, role="holdout", n=n)
        for h in (128, 64) for n, s, k in HOLDOUT
    ]
    diagnostics = [
        make_case(1024, k=128, role="diagnostic"),
        make_case(1024, dataflow="os", role="diagnostic"),
    ]
    pairs = [make_case(s) for s in PAIR_SIZES] + [
        make_case(s, array=64, shard=True) for s in PAIR_SIZES
    ]
    shards = [make_case(s, array=64, shard=True) for s in CALIBRATION]
    extension = [
        make_case(s, k=k, array=h, shard=h == 64, role="diagnostic", n=4096)
        for s, k in [(3968, 128), (2688, 128), (4032, 64)]
        for h in (128, 64)
    ]
    all_groups = {
        "smoke": calibration[:1], "pilot": calibration[:3] + diagnostics,
        "first-holdout": holdout[:1], "holdouts": holdout, "pairs": pairs,
        "remaining": calibration + diagnostics + holdout + shards + pairs + extension,
    }
    return list({run_id(case): case for case in all_groups[group]}.values())


def csv_write(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent,
                                     delete=False) as temp:
        writer = csv.DictWriter(temp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    Path(temp.name).replace(path)


def csv_read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def replace_setting(content, field, value):
    pattern = rf"(?im)^(\s*{re.escape(field)}[ \t]*[:=][ \t]*).*$"
    content, count = re.subn(pattern, lambda match: match[1] + str(value), content)
    if count != 1:
        raise ValueError(f"Expected one {field} setting in base config; found {count}")
    return content


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def report_row(report, field):
    rows = csv_read(report)
    if len(rows) != 1:
        raise ValueError(f"Expected one GEMM in {report}; found {len(rows)} rows")
    columns = {normalized(key): key.strip() for key in rows[0] if key and key.strip()}
    values = {normalized(key): value for key, value in rows[0].items() if key}
    try:
        number = float(values[normalized(field)])
        actual_header = columns[normalized(field)]
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Cannot read {field!r} from {report}. Headers: "
                         f"{', '.join(columns.values())}") from error
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"Invalid cycle count {number} in {report}")
    extra = {}
    for key in ("totalcycles", "computecycles", "stallcycles", "totalcyclesinclstall"):
        if key in values:
            extra["raw_" + key] = values[key].strip()
    return int(number), actual_header, extra


def results(root):
    return [json.loads(path.read_text()) for path in
            sorted((root / "raw_runs").glob("*/result.json"))]


def update_csv(root):
    rows = results(root)
    if rows:
        csv_write(root / "array_runs.csv", rows)


def run_one(case, settings, root, base_text):
    identifier = run_id(case)
    folder = root / "raw_runs" / identifier
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / "array.cfg"
    topology = folder / "gemm.csv"
    output = folder / "reports"
    log_path = folder / "run.log"
    result_path = folder / "result.json"
    rendered = base_text
    for field, value in [("run_name", identifier), ("ArrayHeight", case["array_h"]),
                         ("ArrayWidth", case["array_w"]), ("Dataflow", case["dataflow"])]:
        rendered = replace_setting(rendered, field, value)
    mapping = dict(python=sys.executable, config=str(config), topology=str(topology),
                   output=str(output), adapter=str(Path(__file__).with_name("run_sim.py")))
    command = [part.format_map(mapping) for part in settings["command"]]
    signature = hashlib.sha256(json.dumps([rendered, settings, command],
                                         sort_keys=True).encode()).hexdigest()
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["signature"] != signature:
            raise ValueError(f"Settings changed for {identifier}. Use a new --out folder.")
        if (root / saved["raw_report"]).is_file():
            print(f"SKIP {identifier} ({saved['cycles']} cycles)", flush=True)
            return
        raise ValueError(f"Raw report missing for {identifier}; use a fresh --out folder.")
    if output.exists():
        # Preserve failed or interrupted attempts before retrying.
        attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=folder))
        output.rename(attempt / "reports")
        if log_path.exists():
            log_path.rename(attempt / "run.log")
    config.write_text(rendered)
    topology.write_text(f"Layer Name,M,N,K,\nLU_update,{case['M']},{case['N']},{case['K']},\n")
    output.mkdir()
    print(f"RUN  {identifier}", flush=True)
    with log_path.open("w") as log:
        log.write(f"Command: {command!r}\n\n")
        log.flush()
        subprocess.run(command, cwd=Path(settings["sim_cwd"]).expanduser(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = list(output.rglob("COMPUTE_REPORT.csv"))
    if len(reports) != 1:
        raise ValueError(f"Expected one COMPUTE_REPORT.csv in {output}, found {len(reports)}")
    cycles, field, extra = report_row(reports[0], settings["cycle_column"])
    row = dict(case, run_id=identifier, cycles=cycles, cycle_field=field,
               array_time_us=cycles / 1000, raw_report=str(reports[0].relative_to(root)),
               signature=signature, **extra)
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as temp:
        json.dump(row, temp, indent=2)
    Path(temp.name).replace(result_path)
    print(f"DONE {identifier}: {cycles} cycles ({field})", flush=True)


def run_batch(args):
    source = args.settings.resolve()
    settings = json.loads(source.read_text())
    base = (source.parent / settings["base_config"]).read_text()
    cases = planned_cases(args.group)
    csv_write(args.out / f"manifest_{args.group}.csv",
              [dict(case, run_id=run_id(case)) for case in cases])
    failures = []
    for case in cases:
        try:
            run_one(case, settings, args.out, base)
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            identifier = run_id(case)
            print(f"FAILED {identifier}: {error}\n"
                  f"See {args.out / 'raw_runs' / identifier / 'run.log'}", file=sys.stderr)
            failures.append(identifier)
            if args.group != "remaining":
                break
        update_csv(args.out)
    if failures:
        raise SystemExit(f"{len(failures)} failed run(s). Successful runs remain saved.")


def estimate(case):
    formula = ws_cyc if case["dataflow"] == "ws" else os_cyc
    return formula(*(case[key] for key in ("array_h", "array_w", "M", "N", "K")))


def predict(args):
    cases = planned_cases("remaining")
    rows = [dict(case, run_id=run_id(case), predicted_cycles=estimate(case),
                 model=f"rina_{case['dataflow']}_physical_folds_v1") for case in cases]
    path = args.out / "cycle_predictions.csv"
    if path.exists():
        raise SystemExit(f"{path} already exists. Keep it as your frozen forecast.")
    csv_write(path, rows)
    print(f"Saved {len(rows)} predictions to {path}")


def validate(args):
    predicted = {row["run_id"]: row for row in csv_read(args.out / "cycle_predictions.csv")}
    rows = []
    for row in results(args.out):
        forecast = predicted.get(row["run_id"])
        if forecast is None:
            continue
        expected = int(forecast["predicted_cycles"])
        observed = row["cycles"]
        rows.append({**{key: row[key] for key in KEYS}, "run_id": row["run_id"],
                     "role": forecast["role"], "dataflow": forecast["dataflow"],
                     "model": forecast["model"], "predicted_cycles": expected,
                     "simulated_cycles": observed, "cycle_field": row["cycle_field"],
                     "absolute_percent_error": 100 * abs(expected - observed) / observed})
    csv_write(args.out / "cycle_validation.csv", rows)
    for h in (128, 64):
        check = [row for row in rows if row["role"] == "holdout" and row["array_h"] == h]
        if check:
            errors = [row["absolute_percent_error"] for row in check]
            print(f"{h}x{h}: {len(check)} held-out runs; mean {sum(errors)/len(errors):.2f}%, "
                  f"worst {max(errors):.2f}%")
            for row in check:
                print(f"  {row['run_id']}: {row['predicted_cycles']} predicted vs "
                      f"{row['simulated_cycles']} simulated ({row['absolute_percent_error']:.2f}%)")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(
        (row for row in results(args.out)
         if row["dataflow"] == "ws" and row["layout"] == "single" and row["array_h"] == 128
         and row["K"] == 64 and row["trailing_size"] <= 1984),
        key=lambda row: row["trailing_size"],
    )
    if not rows:
        raise SystemExit("Run the pilot before plotting.")
    fields = {row["cycle_field"] for row in rows}
    if len(fields) != 1:
        raise SystemExit("Plot inputs use different cycle fields; use a consistent field.")
    points = []
    for row in rows:
        utilization = 100 * row["M"] * row["N"] * row["K"] / (16384 * row["cycles"])
        if utilization > 100:
            raise SystemExit(f"{row['run_id']} exceeds 100% utilization. Check the cycle field.")
        points.append(dict(trailing_size=row["trailing_size"], utilization_percent=utilization,
                           cycles=row["cycles"], cycle_field=row["cycle_field"],
                           raw_report=row["raw_report"]))
    csv_write(args.out / "preliminary_figure_data.csv", points)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    ax.scatter([row["trailing_size"] for row in points],
               [row["utilization_percent"] for row in points], color="#245aa2")
    ax.set(xlabel="Trailing matrix dimension (rows or columns)",
           ylabel="Effective utilization (%)", ylim=(0, 105),
           title="Blocked LU updates on one 128 × 128 array")
    ax.grid(axis="y", alpha=.25)
    ax.text(.98, .04, "Block size 64 · Weight-stationary", transform=ax.transAxes,
            ha="right", fontsize=9)
    fig.savefig(args.out / "fig_prelim.pdf")
    plt.close(fig)
    print(f"Saved {len(points)} measured points to {args.out / 'fig_prelim.pdf'}")


def compare(args):
    indexed = {row["run_id"]: row for row in results(args.out)}
    rows = []
    for s in PAIR_SIZES:
        large = indexed.get(run_id(make_case(s)))
        shard = indexed.get(run_id(make_case(s, array=64, shard=True)))
        if large is None or shard is None:
            continue
        if large["cycle_field"] != shard["cycle_field"]:
            raise SystemExit(f"Cycle fields differ for paired update s={s}.")
        macs = s * s * 64
        for idle_cost in (0, .1, .5):
            for replicated_cost in (0, .1, 1):
                def metrics(row, extra_words):
                    cycles = row["cycles"]
                    idle = 16384 * cycles - macs
                    if idle < 0:
                        raise ValueError(f"Negative idle PE-cycles for s={s}")
                    energy = macs + idle_cost * idle + replicated_cost * extra_words
                    return cycles, energy, energy * cycles

                large_cycles, _, large_edp = metrics(large, 0)
                # One shared left operand is delivered to each of four independent arrays.
                for layout, source, extra_words in [
                    ("single", large, 0), ("four", shard, 3 * s * 64)
                ]:
                    cycles, energy, edp = metrics(source, extra_words)
                    rows.append(dict(trailing_size=s, layout=layout, dataflow="ws", cycles=cycles,
                                     useful_macs=macs, extra_input_words=extra_words,
                                     idle_cost_per_pe_cycle=idle_cost,
                                     replication_cost_per_word=replicated_cost,
                                     relative_energy=energy, relative_edp=edp,
                                     normalized_edp=edp / large_edp,
                                     speedup_vs_single=large_cycles / cycles,
                                     raw_report=source["raw_report"]))
    if not rows:
        raise SystemExit("No completed pairs. Run the pairs group first.")
    csv_write(args.out / "layout_comparison.csv", rows)
    print(f"Saved screening estimates to {args.out / 'layout_comparison.csv'}")



def diagnostics(args):
    indexed = {row["run_id"]: row for row in results(args.out)}
    groups = [
        ("Matched OS vs WS (1024,1024,64)", make_case(1024, dataflow="os"),
         make_case(1024)),
        ("WS partial vs full K at fixed M,N", make_case(1024),
         make_case(1024, k=128)),
    ]
    rows = []
    for name, first, second in groups:
        left = indexed.get(run_id(first))
        right = indexed.get(run_id(second))
        if left is None or right is None:
            raise SystemExit("Complete the pilot before running diagnostics.")
        if left["cycle_field"] != right["cycle_field"]:
            raise SystemExit("Pilot inputs use different cycle fields.")
        rows.append(dict(comparison=name, first_run=left["run_id"],
                         second_run=right["run_id"], first_cycles=left["cycles"],
                         second_cycles=right["cycles"], first_over_second=left["cycles"] / right["cycles"],
                         first_predicted=estimate(first), second_predicted=estimate(second),
                         cycle_field=left["cycle_field"]))
    csv_write(args.out / "pilot_diagnostics.csv", rows)
    for row in rows:
        print(f"{row['comparison']}: {row['first_cycles']} vs {row['second_cycles']} cycles "
              f"(ratio {row['first_over_second']:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("work_ws"), help="Results folder")
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Run a batch and export the successful reports")
    run.add_argument("group", choices=["smoke", "pilot", "first-holdout", "holdouts", "pairs", "remaining"])
    run.add_argument("--settings", type=Path, default=Path("settings.json"))
    run.set_defaults(fn=run_batch)
    for name, fn in [("predict", predict), ("validate", validate),
                     ("plot", plot), ("compare", compare), ("diagnostics", diagnostics)]:
        sub.add_parser(name).set_defaults(fn=fn)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.fn(args)


if __name__ == "__main__":
    main()
