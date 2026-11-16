#!/usr/bin/env bash
# Run the default fine-tuned model (C-FT) inside a Docker container capped at
# 2 GiB RAM with no swap, and save the evidence. Run from the repo root after
# `python3 install.py` (needs transfer/finetuned and data/).
#
#   bash scripts/docker_2gb_check.sh [output-folder]
#
# 1. probe: a process that touches 2.5 GiB must be OOM-killed (proves the cap)
# 2. tickets: 3 runs of predict.py on data/tickets.jsonl under the cap
# 3. dev: 1 run on data/dev.jsonl, scored on the host with baseline/evaluate.py
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-evidence/$(date +%Y-%m-%d)-docker-2gb}"
WORK="reports/docker-2gb"              # raw predictions and logs stay local (gitignored)
IMAGE="silab-fa:cft-2gb"
LIMITS=(--memory=2g --memory-swap=2g --cpus=2 --network=none)
mkdir -p "$OUT" "$WORK"
rm -f "$WORK"/DONE "$WORK"/FAILED
trap 'echo "failed at line $LINENO" > "$WORK/FAILED"' ERR

echo "== docker version"; docker version --format '{{.Server.Os}}/{{.Server.Arch}} {{.Server.Version}}' | tee "$WORK/docker_version.txt"
echo "== build"; docker build -q -f docs/testing/Dockerfile.finetuned -t "$IMAGE" . | tee "$WORK/image_id.txt"

MOUNTS=(--mount "type=bind,source=$PWD/transfer/finetuned,target=/model,readonly"
        --mount "type=bind,source=$PWD/data,target=/input,readonly"
        --mount "type=bind,source=$PWD/$WORK,target=/results")

echo "== probe (expect OOM kill)"
docker rm -f silab-probe >/dev/null 2>&1 || true
set +e
docker run --name silab-probe "${LIMITS[@]}" "$IMAGE" Solution/scripts/check_limits.py --probe > "$WORK/probe.log" 2>&1
PROBE_EXIT=$?
set -e
PROBE_OOM=$(docker inspect silab-probe --format '{{.State.OOMKilled}}')
docker inspect silab-probe --format '{{json .HostConfig.Memory}} {{json .HostConfig.MemorySwap}} {{json .HostConfig.NanoCpus}} {{json .HostConfig.NetworkMode}}' > "$WORK/probe_hostconfig.txt"
docker rm silab-probe >/dev/null
echo "probe exit=$PROBE_EXIT oomkilled=$PROBE_OOM"

for i in 1 2 3; do
  echo "== tickets run $i"
  docker run --rm "${LIMITS[@]}" "${MOUNTS[@]}" "$IMAGE" Solution/scripts/check_limits.py \
    --report "/results/limits-$i.json" --run predict.py /input/tickets.jsonl "/results/tickets-$i.jsonl" \
    --method bge-finetuned --model-path /model --metrics "/results/metrics-$i.json" > "$WORK/run-$i.log" 2>&1
done

echo "== dev run"
docker run --rm "${LIMITS[@]}" "${MOUNTS[@]}" "$IMAGE" Solution/scripts/check_limits.py \
  --report /results/limits-dev.json --run predict.py /input/dev.jsonl /results/dev.jsonl \
  --method bge-finetuned --model-path /model > "$WORK/run-dev.log" 2>&1
python3 baseline/evaluate.py "$WORK/dev.jsonl" data/dev.jsonl > "$WORK/dev_eval.txt" 2>&1

python3 - "$WORK" "$OUT" "$PROBE_EXIT" "$PROBE_OOM" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
work, out, probe_exit, probe_oom = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "true"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
runs = []
for i in (1, 2, 3):
    limits = json.loads((work / f"limits-{i}.json").read_text())
    metrics = json.loads((work / f"metrics-{i}.json").read_text())
    lat = metrics["latency"]
    runs.append({"run": i, "exit_code": limits["exit_code"],
                 "memory.max": limits["memory.max"], "memory.swap.max": limits["memory.swap.max"],
                 "cpu.max": limits["cpu.max"],
                 "memory_peak_mib": round(int(limits["memory.peak"]) / 2**20, 1),
                 "oom_events_after": limits["memory.events_after"],
                 "process_wall_seconds": round(limits["process_wall_seconds"], 3),
                 "p50_ms": round(lat["p50_seconds"] * 1000, 3), "p95_ms": round(lat["p95_seconds"] * 1000, 3),
                 "max_ms": round(lat["max_seconds"] * 1000, 3),
                 "rows": sum(1 for _ in open(work / f"tickets-{i}.jsonl")),
                 "output_sha256": sha(work / f"tickets-{i}.jsonl"),
                 "per_ticket_seconds": [[t["ticket_id"], round(t["seconds"], 6)] for t in metrics["tickets"]]})
dev_limits = json.loads((work / "limits-dev.json").read_text())
scores = [l for l in (work / "dev_eval.txt").read_text(encoding="utf-8").splitlines()
          if l.split() and l.split()[0] in ("device_id", "category", "urgency", "duplicate_of", "總分")]
deliverable = Path("tickets_predictions.jsonl")
summary = {
    "docker": (work / "docker_version.txt").read_text().strip(),
    "image_id": (work / "image_id.txt").read_text().strip(),
    "limits": "--memory=2g --memory-swap=2g --cpus=2 --network=none",
    "probe": {"exit_code": probe_exit, "oom_killed": probe_oom,
              "host_config": (work / "probe_hostconfig.txt").read_text().strip()},
    "tickets_runs": [{k: v for k, v in r.items() if k != "per_ticket_seconds"} for r in runs],
    "identical_outputs": len({r["output_sha256"] for r in runs}) == 1,
    "matches_tickets_predictions_jsonl": deliverable.is_file() and runs[0]["output_sha256"] == sha(deliverable),
    "dev": {"exit_code": dev_limits["exit_code"], "memory_peak_mib": round(int(dev_limits["memory.peak"]) / 2**20, 1),
            "score_lines": scores},
}
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
ok = (probe_oom and all(r["exit_code"] == 0 and r["rows"] == 2000 for r in runs) and dev_limits["exit_code"] == 0)
print(json.dumps({k: v for k, v in summary.items() if k != "tickets_runs"}, ensure_ascii=False, indent=2))
for r in summary["tickets_runs"]:
    print(f"run {r['run']}: exit {r['exit_code']}, {r['process_wall_seconds']} s, peak {r['memory_peak_mib']} MiB, P95 {r['p95_ms']} ms")
print("RESULT:", "PASS" if ok else "FAIL")
PY
touch "$WORK/DONE"
