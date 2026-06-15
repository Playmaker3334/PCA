import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ACTIVATIONS, LOG_DIR, METRICS, PC_DIR, RAW, get_logger, load_json

logger = get_logger("pipeline")

SCRIPTS_DIR = Path(__file__).resolve().parent

STAGES = [
    ("00_download",   "00_download.py",            RAW / "alpaca"),
    ("01_extract",    "01_extract_activations.py", None),
    ("02_localize",   "02_localize_features.py",   METRICS / "localization_decision.json"),
    ("03_train_pc",   "03_train_pc.py",            PC_DIR / "monitor" / "config.json"),
    ("04_evaluate",   "04_evaluate.py",            METRICS / "evaluation_summary.json"),
    ("05_baselines",  "05_run_baselines.py",       METRICS / "baselines.json"),
    ("07_density",    "07_density_baselines.py",   METRICS / "density_baselines.json"),
]

CRITICAL = {"00_download", "01_extract", "02_localize", "03_train_pc"}

EXPECTED_DATASETS = ["alpaca", "triviaqa", "mmlu", "gsm8k",
                     "jailbreakbench", "harmful_hirundo", "xstest"]


def stage_done(stage_id, sentinel):
    if stage_id == "00_download":
        return all((RAW / n).exists() and any((RAW / n).iterdir())
                   for n in EXPECTED_DATASETS)
    if stage_id == "01_extract":
        return all((ACTIVATIONS / f"{n}.npz").exists() for n in EXPECTED_DATASETS)
    if sentinel is None:
        return False
    return sentinel.exists()


def run_stage(stage_id, script):
    script_path = SCRIPTS_DIR / script
    logger.info(f"=== START {stage_id} ({script}) ===")
    t0 = time.time()
    stage_log = LOG_DIR / f"stage_{stage_id}.log"
    with open(stage_log, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            lf.flush()
        proc.wait()
    dt = time.time() - t0
    ok = proc.returncode == 0
    logger.info(f"=== END {stage_id} rc={proc.returncode} ({dt:.1f}s) ===")
    return {"stage": stage_id, "returncode": proc.returncode, "ok": ok,
            "duration_s": round(dt, 1), "log": str(stage_log)}


def consolidate_csv():
    import csv
    rows = []

    def add(metric, value, source):
        rows.append({"metric": metric, "value": value, "source": source})

    try:
        ev = load_json(METRICS / "evaluation_summary.json")
        for k, v in ev.items():
            add(k, v, "evaluation_summary")
    except Exception:
        pass

    try:
        db = load_json(METRICS / "density_baselines.json")
        for k, v in db.items():
            if isinstance(v, dict) and "auroc" in v:
                add(f"density_auroc__{k}", v["auroc"], "density_baselines")
    except Exception:
        pass

    try:
        bl = load_json(METRICS / "baselines.json")
        if bl.get("refusal_direction"):
            add("refusal_direction_auroc", bl["refusal_direction"]["auroc"], "baselines")
    except Exception:
        pass

    out = METRICS / "results_master.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "source"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"consolidated {len(rows)} metrics -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_stage", default=None)
    ap.add_argument("--to", dest="to_stage", default=None)
    ap.add_argument("--only", dest="only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ids = [s[0] for s in STAGES]
    selected = STAGES
    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    else:
        if args.from_stage:
            i = ids.index(args.from_stage)
            selected = [s for s in selected if ids.index(s[0]) >= i]
        if args.to_stage:
            j = ids.index(args.to_stage)
            selected = [s for s in selected if ids.index(s[0]) <= j]

    status = {"started": datetime.now().isoformat(), "stages": []}
    logger.info(f"pipeline plan: {[s[0] for s in selected]}")

    for stage_id, script, sentinel in selected:
        if not args.force and stage_done(stage_id, sentinel):
            logger.info(f"--- SKIP {stage_id} (artefacto existe) ---")
            status["stages"].append({"stage": stage_id, "skipped": True, "ok": True})
            continue

        res = run_stage(stage_id, script)
        status["stages"].append(res)

        status["updated"] = datetime.now().isoformat()
        with open(LOG_DIR / "pipeline_status.json", "w") as f:
            json.dump(status, f, indent=2)

        if not res["ok"]:
            if stage_id in CRITICAL:
                logger.error(f"etapa critica {stage_id} fallo. abortando.")
                break
            else:
                logger.warning(f"etapa independiente {stage_id} fallo. continuando.")

    csv_path = consolidate_csv()
    status["finished"] = datetime.now().isoformat()
    status["results_csv"] = str(csv_path)
    with open(LOG_DIR / "pipeline_status.json", "w") as f:
        json.dump(status, f, indent=2)

    logger.info("=== PIPELINE DONE ===")
    logger.info(f"status: {LOG_DIR / 'pipeline_status.json'}")
    logger.info(f"results CSV: {csv_path}")


if __name__ == "__main__":
    main()