"""「一次跑全部方法」背景工作：依序執行、單一失敗不中斷、結果可回查。"""
import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch

from helper_tool import batch_runs


def fake_result(method, total):
    summary = {"total": total, "metrics": {"category": total, "device": .9, "urgency": .8, "duplicate": .7}}
    return {"summary": summary, "errors": [1, 2], "records": [], "label": f"{method}.jsonl",
            "run_id": f"run-{method}", "downloads": {},
            "execution": {"output_path": f"/out/{method}.jsonl", "count": 3},
            "timing": {"total_seconds": 1.0, "prediction_seconds": .5}}


def wait(job_id):
    for _ in range(200):
        job = batch_runs.get_job(job_id)
        if job["state"] != "running":
            return job
        time.sleep(.01)
    raise AssertionError("背景工作未結束")


REAL_MODEL_CHOICES = batch_runs.model_choices  # captured before setUp patches it
EMPTY = {"choices": {m: [] for m in batch_runs.MODEL_METHODS}, "defaults": batch_runs.DEFAULT_PATHS}


def training_job(job_id, kind, results, linear=None):
    config = {"kind": kind, "linear_settings": linear}
    return {"job_id": job_id, "state": "succeeded", "config": config, "comparison": {"results": results}}


class BatchRunsTest(unittest.TestCase):
    def setUp(self):
        for store in (batch_runs._jobs, batch_runs._results):
            store.clear()
        batch_runs._order.clear()
        patcher = patch.object(batch_runs, "model_choices", return_value=EMPTY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runs_every_method_sequentially_and_survives_failures(self):
        calls = []

        def run(payload, reports):
            calls.append(payload)
            if payload["method"] == "tfidf-svc":
                raise ValueError("找不到模型")
            return fake_result(payload["method"], {"baseline": .44, "solution-a": .9}.get(payload["method"], .6))

        missing = {**batch_runs.DEFAULT_PATHS, "bge-finetuned": "does-not-exist"}
        with patch.object(batch_runs, "run_experiment", side_effect=run), \
                patch.object(batch_runs, "DEFAULT_PATHS", missing):
            job = batch_runs.start_job({"dataset": "history", "limit": "native", "output_dir": "x",
                                        "method": "ignored", "model_path": "ignored"}, None, threading.Lock())
            job = wait(job["job_id"])
        states = {i["method"]: i["state"] for i in job["items"]}
        self.assertEqual(states["tfidf-svc"], "failed")
        self.assertEqual(states["bge-finetuned"], "skipped")
        self.assertEqual(sum(s == "succeeded" for s in states.values()), 5)
        self.assertEqual([c["method"] for c in calls], list(batch_runs.ORDER[:-1]))
        self.assertTrue(all(c["dataset"] == "history" and c["model_path"] == "" for c in calls))
        solution = next(i for i in job["items"] if i["method"] == "solution-a")
        self.assertEqual((solution["total"], solution["error_count"]), (.9, 2))
        self.assertEqual(batch_runs.get_result(job["job_id"], "solution-a")["run_id"], "run-solution-a")
        self.assertEqual(batch_runs.get_job()["job_id"], job["job_id"])
        with self.assertRaises(ValueError):
            batch_runs.get_result(job["job_id"], "tfidf-svc")

    def test_rejects_concurrent_jobs_and_bad_input(self):
        release = threading.Event()

        def run(payload, reports):
            release.wait(2)
            return fake_result(payload["method"], .5)

        with patch.object(batch_runs, "run_experiment", side_effect=run):
            job = batch_runs.start_job({"dataset": "history", "methods": ["baseline"]}, None, threading.Lock())
            with self.assertRaises(ValueError):
                batch_runs.start_job({"dataset": "history"}, None, threading.Lock())
            release.set()
            wait(job["job_id"])
        for payload in ({"dataset": "secret"}, {"limit": "8gb"}, {"methods": ["rm -rf"]}):
            with self.assertRaises(ValueError):
                batch_runs.start_job(payload, None, threading.Lock())

    def test_model_choices_list_training_runs_newest_first(self):
        folder = Path(tempfile.mkdtemp())
        paths = {name: folder / name for name in ("old.joblib", "new-raw.joblib", "new-masked.joblib", "ft", "gone.joblib")}
        for name, path in paths.items():
            if name != "gone.joblib":
                path.mkdir() if name == "ft" else path.write_text("x")
        jobs = [
            training_job("20261116-153913-1-a", "tfidf", [
                {"variant": "tfidf-lr-raw", "method": "tfidf-lr", "model_path": str(paths["new-raw.joblib"]),
                 "pipeline_total": .81, "selected": False},
                {"variant": "tfidf-lr-masked", "method": "tfidf-lr", "model_path": str(paths["new-masked.joblib"]),
                 "pipeline_total": .83, "selected": True},
                {"variant": "tfidf-svc-raw", "method": "tfidf-svc", "model_path": str(paths["gone.joblib"])}],
                {**batch_runs.LINEAR_DEFAULTS, "c": .5}),
            training_job("20261116-152656-1-b", "tfidf", [
                {"variant": "tfidf-lr-raw", "method": "tfidf-lr", "model_path": str(paths["old.joblib"]),
                 "selected": True}]),
            training_job("20261116-150000-1-c", "bge-finetune", [
                {"variant": "finetuned", "model_path": str(paths["ft"]), "pipeline_total": .9}]),
            {**training_job("20261116-160000-1-d", "tfidf", []), "state": "failed"},
        ]
        jobs[2]["config"]["settings"] = {"epochs": 5, "encoder_lr": 1e-5}
        result = REAL_MODEL_CHOICES(jobs)
        lr = result["choices"]["tfidf-lr"]
        self.assertEqual([i["path"] for i in lr], [str(paths["new-masked.joblib"]), str(paths["new-raw.joblib"]),
                                                    str(paths["old.joblib"])])
        self.assertIn("★ 選型結果", lr[0]["label"])
        self.assertIn("11/16 15:39", lr[0]["label"])
        self.assertIn("c=0.5", lr[0]["label"])
        self.assertIn("驗證總分 0.8300", lr[0]["label"])
        self.assertIn("預設參數（舊紀錄）", lr[2]["label"])
        self.assertEqual(result["choices"]["tfidf-svc"], [], "已刪除的模型檔不列出")
        self.assertEqual(result["choices"]["bge-finetuned"][0]["path"], str(paths["ft"]))
        self.assertIn("epochs=5", result["choices"]["bge-finetuned"][0]["label"])

        self.assertEqual(batch_runs.resolve_model("tfidf-lr", "latest", result)[0], str(paths["new-masked.joblib"]))
        self.assertEqual(batch_runs.resolve_model("tfidf-lr", None, result)[0], str(paths["new-masked.joblib"]))
        self.assertEqual(batch_runs.resolve_model("tfidf-lr", str(paths["old.joblib"]), result)[0], str(paths["old.joblib"]))
        self.assertEqual(batch_runs.resolve_model("tfidf-lr", "default", result), ("", "預設位置：Solution/models/tfidf-lr.joblib"))
        path, label = batch_runs.resolve_model("bge-lr", "latest", result)
        self.assertEqual(path, "")
        self.assertIn("沒有訓練紀錄", label)
        self.assertTrue(batch_runs.resolve_model("bge-finetuned", "default", result)[0].endswith("transfer/finetuned"))
        for choice in (str(paths["gone.joblib"]), "/etc/passwd"):
            with self.assertRaisesRegex(ValueError, "不在訓練紀錄"):
                batch_runs.resolve_model("tfidf-lr", choice, result)

    def test_selected_models_reach_each_experiment(self):
        choices = {"choices": {**EMPTY["choices"], "tfidf-lr": [{"path": "/runs/a/tfidf-lr.joblib", "label": "A"}]},
                   "defaults": batch_runs.DEFAULT_PATHS}
        calls = []
        with patch.object(batch_runs, "model_choices", return_value=choices), \
                patch.object(batch_runs, "run_experiment",
                             side_effect=lambda payload, reports: calls.append(payload) or fake_result(payload["method"], .5)):
            job = wait(batch_runs.start_job({"dataset": "history", "methods": ["tfidf-lr", "tfidf-svc"],
                                             "models": {"tfidf-lr": "/runs/a/tfidf-lr.joblib", "tfidf-svc": "default"}},
                                            None, threading.Lock())["job_id"])
        self.assertEqual([c["model_path"] for c in calls], ["/runs/a/tfidf-lr.joblib", ""])
        self.assertEqual(job["items"][0]["model_label"], "A")
        self.assertIn("預設位置", job["items"][1]["model_label"])
        with patch.object(batch_runs, "model_choices", return_value=choices):
            for models in ({"tfidf-lr": "/runs/deleted.joblib"}, {"baseline": "x"}, ["latest"]):
                with self.subTest(models=models), self.assertRaises(ValueError):
                    batch_runs.start_job({"dataset": "history", "methods": ["tfidf-lr"], "models": models},
                                         None, threading.Lock())

    def test_unscored_dataset_keeps_none(self):
        result = fake_result("baseline", .5)
        result["summary"] = None
        with patch.object(batch_runs, "run_experiment", return_value=result):
            job = wait(batch_runs.start_job({"dataset": "tickets", "methods": ["baseline"]}, None,
                                            threading.Lock())["job_id"])
        self.assertIsNone(job["items"][0]["total"])
        self.assertIsNone(job["items"][0]["metrics"])


if __name__ == "__main__":
    unittest.main()
