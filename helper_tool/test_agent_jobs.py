"""Agent UI job contracts using synthetic subprocesses; no dev or LLM reads."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from helper_tool import agent_jobs as jobs, ui_server

JOB_ID = "20261114-120000-123456-abcdef12"


class AgentJobsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.registry = self.root / "registry"
        registry = patch.object(jobs, "REGISTRY", self.registry)
        registry.start()
        self.addCleanup(registry.stop)

    def seed(self, state="queued", pid=None):
        folder = jobs.directory(JOB_ID)
        folder.mkdir(parents=True)
        job = {"job_id": JOB_ID, "kind": "synthetic", "threads": 6, "state": state,
               "created_at": "2026-01-01T00:00:00+00:00", "pid": pid, "message": "synthetic",
               "total": 2, "completed": 0, "passed": 0, "cases": [], "timing": {}}
        jobs.save(folder / "status.json", job)
        return folder

    def test_invalid_payloads_rejected_before_opening_sources(self):
        for payload in ({"kind": "dev"}, {"kind": []}, {"kind": "synthetic", "threads": True},
                        {"kind": "history", "threads": 0}, {"kind": "patrol", "command": "anything"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                jobs.start_job(payload)
        self.assertFalse(self.registry.exists())

    def test_missing_assets_is_actionable_and_creates_no_job(self):
        with patch.object(jobs, "availability", return_value={"ready": False}):
            with self.assertRaisesRegex(ValueError, "setup.py"):
                jobs.start_job({"kind": "synthetic"})
        self.assertFalse(self.registry.exists())

    def test_background_job_is_persisted_and_second_run_rejected(self):
        process = Mock(pid=os.getpid())
        with patch.object(jobs, "availability", return_value={"ready": True}), patch.object(jobs.subprocess, "Popen", return_value=process) as launch:
            result = jobs.start_job({"kind": "synthetic", "threads": 4})
            self.assertEqual(result["state"], "queued")
            self.assertIn("helper_tool.agent_jobs", launch.call_args.args[0])
            self.assertEqual(jobs.list_jobs()[0]["job_id"], result["job_id"])
            with self.assertRaisesRegex(ValueError, "已有"):
                jobs.start_job({"kind": "patrol"})

    def test_dead_worker_becomes_interrupted_without_losing_cases(self):
        folder = self.seed("running", 123456)
        with patch.object(jobs, "alive", return_value=False):
            result = jobs.get_job(JOB_ID)
        self.assertEqual(result["state"], "interrupted")
        self.assertEqual(json.loads((folder / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")

    def test_job_ids_cannot_escape_report_directory(self):
        for value in (None, "../report", JOB_ID + "/../anything"):
            with self.assertRaises(ValueError):
                jobs.directory(value)

    def fake_runner(self, tail):
        (self.root / "evals").mkdir()
        body = '''import json,sys
from pathlib import Path
output=Path(sys.argv[sys.argv.index('--output')+1])
case={'id':'synthetic-one','question':'合成題','expected':2,'actual':{'answer':1,'evidence':[]},'passed':False,'elapsed_s':0.001,'exit_code':0}
print(json.dumps({'event':'started','total':1}),flush=True)
print(json.dumps({'event':'case','case':case}),flush=True)
'''
        (self.root / "evals/run.py").write_text(body + tail, encoding="utf-8")

    def test_wrong_answer_exit_one_is_completed_not_infrastructure_failure(self):
        self.seed()
        self.fake_runner("output.write_text(json.dumps({'cases':[case],'passed':0,'total':1}),encoding='utf-8')\nsys.exit(1)\n")
        with patch.object(jobs, "ROOT", self.root):
            jobs.worker(JOB_ID)
        result = jobs.get_job(JOB_ID)
        self.assertEqual(result["state"], "completed")
        self.assertEqual((result["completed"], result["passed"]), (1, 0))
        self.assertEqual(result["cases"][0]["question"], "合成題")
        self.assertEqual(result["timing"]["test_seconds"], .001)
        self.assertGreater(result["timing"]["total_seconds"], .001)
        self.assertIsNotNone(result["report_url"])

    def test_crash_retains_partial_result_and_error_time(self):
        self.seed()
        self.fake_runner("raise RuntimeError('synthetic worker crash')\n")
        with patch.object(jobs, "ROOT", self.root):
            jobs.worker(JOB_ID)
        result = jobs.get_job(JOB_ID)
        self.assertEqual(result["state"], "failed")
        self.assertIn("synthetic worker crash", result["message"])
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["timing"]["test_seconds"], .001)

    def test_deadline_kills_stuck_runner(self):
        self.seed()
        (self.root / "evals").mkdir()
        (self.root / "evals/run.py").write_text("import time; time.sleep(60)", encoding="utf-8")
        real_timer = threading.Timer
        started = time.monotonic()
        with patch.object(jobs, "ROOT", self.root), patch.object(jobs.threading, "Timer", side_effect=lambda delay, callback: real_timer(.2, callback)):
            jobs.worker(JOB_ID)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(jobs.get_job(JOB_ID)["state"], "failed")


class AgentHTTPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.object(jobs, "REGISTRY", Path(self.temp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.server = ui_server.LocalHTTPServer(("127.0.0.1", 0), ui_server.Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def test_agent_page_and_poll_endpoint_are_available(self):
        with urlopen(self.url + "/") as response:
            self.assertIn(b'id="page-agent"', response.read())
        with urlopen(self.url + "/api/agent/jobs") as response:
            result = json.load(response)
        self.assertEqual(result["jobs"], [])
        self.assertIn("ready", result["availability"])

    def test_dev_and_cross_origin_requests_are_rejected(self):
        for origin, expected in ((None, 400), ("https://example.invalid", 403)):
            headers = {"Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            request = Request(self.url + "/api/agent/jobs", data=b'{"kind":"dev"}', headers=headers)
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, expected)

    def test_report_download_is_scoped_to_valid_job(self):
        folder = jobs.directory(JOB_ID)
        folder.mkdir()
        (folder / "report.json").write_text('{"passed":1}', encoding="utf-8")
        with urlopen(self.url + "/api/agent/report?id=" + JOB_ID) as response:
            self.assertEqual(json.load(response), {"passed": 1})
            self.assertIn("attachment", response.headers["Content-Disposition"])
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + "/api/agent/report?id=../report")
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
