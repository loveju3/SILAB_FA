#!/usr/bin/env python3
"""Measure Part 1 time and memory on the full tickets file and save auditable evidence.

    python3 scripts/collect_evidence.py --runs 3 --output evidence/<name>

For each method (default bge-finetuned, rules) it starts `runs` fresh processes of
`predict.py data/tickets.jsonl ...` with no warm-up and records, per run:
external wall time, model/load time, per-ticket latency (P50/P95/max and every
ticket's seconds), peak RSS, exit code and the output's SHA-256.

On Linux it repeats the runs under a hard 2 GiB address-space limit (RLIMIT_AS,
set in the child before exec). Address space is always >= resident memory, so a
run that finishes under it never held more than 2 GiB. A probe that tries to
allocate 2.5 GiB under the same limit must fail, showing the limit is enforced.
The 2 GB deliverable command itself (`--mem-limit 2g`) is what is run for scheme A.

Nothing from the tickets except ticket_id is written; dev is never read.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _venv  # noqa: E402

LIMIT = 2 * 1024 ** 3
PROBE_BYTES = int(2.5 * 1024 ** 3)
DELIVERABLES = {("bge-finetuned", False): "tickets_predictions.jsonl",
                ("rules", True): "tickets_predictions_2gb.jsonl"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def limit_address_space():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))


def memory_total_bytes():
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    except (OSError, ValueError):
        pass
    return None


def environment():
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    model = ROOT / "transfer/finetuned/model.json"
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "memory_total_bytes": memory_total_bytes(),
        "python": sys.version.split()[0], "git_head": git("rev-parse", "HEAD"),
        "git_dirty_files": [l[3:] for l in git("status", "--porcelain").splitlines() if not l.startswith("??")],
        "tickets_sha256": sha256(ROOT / "data/tickets.jsonl"),
        "finetuned_heads_sha256": json.loads(model.read_text(encoding="utf-8")).get("heads_sha256") if model.is_file() else None,
        "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def run_once(method, limited, workdir, index):
    output = workdir / f"{method}{'-2gb' if limited else ''}-{index}.jsonl"
    metrics = output.with_suffix(".metrics.json")
    command = [sys.executable, str(ROOT / "predict.py"), str(ROOT / "data/tickets.jsonl"), str(output),
               "--metrics", str(metrics)]
    command += ["--mem-limit", "2g"] if (limited and method == "rules") else ["--method", method]
    env = dict(os.environ, SILAB_NO_VENV="1", PYTHONUTF8="1")
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, env=env,
                             preexec_fn=limit_address_space if limited else None)
    wall = time.perf_counter() - started
    record = {"run": index, "method": method, "limit": "RLIMIT_AS 2 GiB" if limited else None,
              "command": ["python", "predict.py", "data/tickets.jsonl", "<out>", *command[5:7], *command[7:]],
              "exit_code": process.returncode, "external_wall_seconds": round(wall, 3),
              "stderr_tail": process.stderr.strip().splitlines()[-3:]}
    record["command"] = ["python", "predict.py", "data/tickets.jsonl", "<out>"] + [
        part if not part.startswith(str(workdir)) else "<metrics>" for part in command[4:]]
    if process.returncode == 0:
        data = json.loads(metrics.read_text(encoding="utf-8"))
        latency = data["latency"]
        record.update({
            "rows": sum(1 for _ in output.open(encoding="utf-8")),
            "output_sha256": sha256(output),
            "load_seconds": round(data["load_seconds"], 3),
            "elapsed_to_output_seconds": round(data["elapsed_to_output_seconds"], 3),
            "peak_rss_mib": round(data["peak_rss_bytes"] / 1024 ** 2, 1),
            "p50_ms": round(latency["p50_seconds"] * 1000, 3),
            "p95_ms": round(latency["p95_seconds"] * 1000, 3),
            "max_ms": round(latency["max_seconds"] * 1000, 3),
            "mean_ms_per_ticket_total": round(wall / latency["count"] * 1000, 3),
            "latency_definition": latency.get("method"),
            "per_ticket_seconds": [[t["ticket_id"], round(t["seconds"], 6)] for t in data["tickets"]],
        })
    return record


def probe():
    code = f"b = bytearray({PROBE_BYTES}); print('allocated')"
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             preexec_fn=limit_address_space)
    readback = subprocess.run([sys.executable, "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_AS))"],
                              capture_output=True, text=True, preexec_fn=limit_address_space).stdout.strip()
    return {"allocation_bytes": PROBE_BYTES, "exit_code": process.returncode,
            "blocked": process.returncode != 0 and "allocated" not in process.stdout,
            "error_tail": process.stderr.strip().splitlines()[-1:], "rlimit_readback": readback}


def summarize(records):
    ok = [r for r in records if r["exit_code"] == 0]
    span = lambda key: [min(r[key] for r in ok), max(r[key] for r in ok)] if ok else None
    return {"runs": len(records), "passed": len(ok),
            "external_wall_seconds": span("external_wall_seconds"), "p50_ms": span("p50_ms"),
            "p95_ms": span("p95_ms"), "max_ms": span("max_ms"), "peak_rss_mib": span("peak_rss_mib"),
            "identical_outputs": len({r["output_sha256"] for r in ok}) == 1 if ok else None,
            "within_480s": all(r["external_wall_seconds"] <= 480 for r in ok) and len(ok) == len(records)}


def fmt(span, digits=1):
    if not span:
        return "—"
    low, high = f"{span[0]:.{digits}f}", f"{span[1]:.{digits}f}"
    return low if low == high else f"{low}～{high}"


def write_report(folder, env, groups, probe_result):
    lines = ["# Part 1 量測證據", "",
             f"由 `scripts/collect_evidence.py` 產生，{env['measured_at_utc']}。每列為新 Python 程序、無預熱、不清 OS 快取；"
             "總時間由外部 `perf_counter()` 量整個程序（含 Python 啟動、模型載入、推論、寫檔）。"
             "逐筆時間的定義見各 `runs/*.json` 的 `latency_definition`。", "",
             f"- 環境：{env['platform']}，{env['cpu_count']} CPU，"
             f"RAM {round(env['memory_total_bytes'] / 1024 ** 3, 1) if env['memory_total_bytes'] else '?'} GiB，Python {env['python']}",
             f"- 程式版本：`{env['git_head']}`" + ("（工作目錄有未提交修改）" if env["git_dirty_files"] else "（乾淨）"),
             f"- 輸入：data/tickets.jsonl，SHA-256 `{env['tickets_sha256'][:16]}…`", "",
             "| 方法 | 記憶體限制 | 通過 | 總時間（秒） | 平均每筆（ms） | P50（ms） | P95（ms） | 最大（ms） | 峰值 RSS（MiB） | 三次輸出相同 | 與 repo 交付檔相同 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for (method, limited), data in groups.items():
        s = data["summary"]
        mean = [r["mean_ms_per_ticket_total"] for r in data["runs"] if r["exit_code"] == 0]
        lines.append(f"| {method} | {'RLIMIT_AS 2 GiB' if limited else '無'} | {s['passed']}/{s['runs']} | "
                     f"{fmt(s['external_wall_seconds'], 2)} | {fmt([min(mean), max(mean)] if mean else None, 1)} | "
                     f"{fmt(s['p50_ms'])} | {fmt(s['p95_ms'])} | {fmt(s['max_ms'])} | {fmt(s['peak_rss_mib'])} | "
                     f"{'是' if s['identical_outputs'] else '否'} | {data.get('matches_deliverable', '—')} |")
    lines += ["", "題目門檻：2,000 筆總時間 ≤ 480 秒（平均每筆 ≤ 240 ms）；2 GB 版本須在 2 GB 上限下完成。", ""]
    if probe_result:
        lines += ["## 2 GiB 限制確實生效",
                  "",
                  f"在相同 RLIMIT_AS 下嘗試配置 {probe_result['allocation_bytes'] / 1024 ** 3:.1f} GiB："
                  f"{'被拒絕' if probe_result['blocked'] else '未被拒絕（限制無效）'}（exit {probe_result['exit_code']}，"
                  f"`{' '.join(probe_result['error_tail'])}`）；子程序讀回的限制為 `{probe_result['rlimit_readback']}`。",
                  "",
                  "RLIMIT_AS 限制的是位址空間，永遠 ≥ 實際佔用的記憶體，所以在它之下跑完代表實際記憶體從未超過 2 GiB；"
                  "這比 Docker 的 cgroup 限制更嚴格。Docker cgroup v2 的方案 A 驗證見 [Solution/budget.md](../../Solution/budget.md)。", ""]
    lines += ["## 檔案", "", "- `summary.json`：環境、各組彙總與限制探測。",
              "- `runs/<方法>.json`：每次執行的完整紀錄，含 2,000 筆逐筆秒數（只有 ticket_id，無工單內容）。", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=["bge-finetuned", "rules"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (ROOT / "data/tickets.jsonl").is_file():
        parser.error("需要 data/tickets.jsonl")
    linux = sys.platform.startswith("linux")
    folder = args.output if args.output.is_absolute() else ROOT / args.output
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    env, groups = environment(), {}
    with tempfile.TemporaryDirectory() as temporary:
        workdir = Path(temporary)
        for method in args.methods:
            for limited in ((False, True) if linux else (False,)):
                print(f"== {method}{' (2 GiB)' if limited else ''}", flush=True)
                runs = []
                for index in range(1, args.runs + 1):
                    record = run_once(method, limited, workdir, index)
                    runs.append(record)
                    print(f"  run {index}: exit {record['exit_code']}, {record['external_wall_seconds']} s", flush=True)
                data = {"summary": summarize(runs), "runs": runs}
                deliverable = DELIVERABLES.get((method, limited))
                if deliverable and (ROOT / deliverable).is_file() and runs[0]["exit_code"] == 0:
                    data["matches_deliverable"] = "是" if runs[0]["output_sha256"] == sha256(ROOT / deliverable) else "否"
                groups[(method, limited)] = data
                name = f"{method}{'-2gib' if limited else ''}.json"
                (folder / "runs" / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # The report covers every group saved in this folder, so methods can be measured
    # in separate invocations (e.g. --methods bge-finetuned, then --methods rules).
    for path in sorted((folder / "runs").glob("*.json")):
        key = (path.stem[:-5], True) if path.stem.endswith("-2gib") else (path.stem, False)
        groups.setdefault(key, json.loads(path.read_text(encoding="utf-8")))
    order = {m: i for i, m in enumerate(["bge-finetuned", "rules", "tfidf-lr", "tfidf-svc", "bge-lr", "bge-svc"])}
    groups = dict(sorted(groups.items(), key=lambda item: (order.get(item[0][0], 9), item[0][1])))
    probe_result = probe() if linux else None
    summary = {"environment": env, "probe_2gib": probe_result,
               "groups": {f"{m}{' 2GiB' if l else ''}": {**d["summary"], "matches_deliverable": d.get("matches_deliverable")}
                          for (m, l), d in groups.items()}}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(folder, env, groups, probe_result)
    print((folder / "README.md").read_text(encoding="utf-8"))
    failed = any(not d["summary"]["within_480s"] for d in groups.values()) or (probe_result and not probe_result["blocked"])
    return 1 if failed else 0


if __name__ == "__main__":
    _venv.ensure()
    sys.exit(main())
