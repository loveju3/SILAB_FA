import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest

from agent.core import Agent
from agent.tools import ROOT, Store
from agent.process import ChildGuard
from patrol import shortlist, selection_output, rank_candidates, prioritize_candidates


class ScriptedModel:
    """Only tests orchestration; never counts as an LLM accuracy result."""
    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, messages, schema):
        return next(self.actions)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT / "evals/fixtures.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "synthetic.jsonl"
        self.path.write_text("".join(json.dumps(r) + "\n" for r in self.fixture["tickets"]), encoding="utf-8")
        self.store = Store(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def run_agent(self, actions, **kwargs):
        return Agent(self.store, ScriptedModel(actions), time.monotonic() + 3, **kwargs).ask("合成測試")

    def test_public_lookup_does_not_read_tickets(self):
        store = Store(Path(self.temp.name) / "nonexistent.jsonl")
        self.assertEqual(store.call("devices_at_location", {"location": "B1 動力室"})["answer"], ["D01", "D06"])

    def test_device_single_and_ambiguous(self):
        self.assertEqual(self.store.call("get_device", {"identifier": "D11", "field": "name"})["answer"], "飲水機")
        with self.assertRaises(ValueError):
            self.store.get_devices(location="頂樓", single=True)

    def test_error_code_variants_and_unknown(self):
        self.assertEqual(self.store.call("get_error_code", {"code": "ｅ＿０３", "field": "meaning"})["answer"], "冷媒低壓警報")
        self.assertIsNone(self.store.call("get_error_code", {"code": "E-999"})["answer"])

    def test_time_boundary_includes_equivalent_timezone(self):
        result = self.store.call("query_tickets", {"operation": "count", "since": "2026-10-01T00:00:00Z",
                                                  "until": "2026-10-02T00:00:00Z"})
        self.assertEqual(result["answer"], 2)
        self.assertEqual(result["evidence"], ["SYN002", "SYN003"])

    def test_invalid_time_and_inverted_interval(self):
        for args in ({"since": "2026-10-01"}, {"since": "2026-10-02T00:00:00Z", "until": "2026-10-01T00:00:00Z"}):
            with self.assertRaises(ValueError):
                self.store.call("query_tickets", {"operation": "count", **args})

    def test_complete_aggregation_and_evidence_cap(self):
        self.path.write_text("".join(json.dumps({**self.fixture["tickets"][0], "ticket_id": f"S{i}"}) + "\n" for i in range(31)), encoding="utf-8")
        result = self.store.call("query_tickets", {"operation": "count"})
        self.assertEqual(result["answer"], 31)
        self.assertEqual(len(result["evidence"]), 20)
        read = self.store.call("query_tickets", {"operation": "read"})
        self.assertTrue(read["truncated"])
        self.assertEqual(len(read["records"]), 20)

    def test_top_reporter_uses_all_tickets_and_ties(self):
        self.assertEqual(self.store.call("query_tickets", {"operation": "top_reporter"})["answer"], "測試乙")
        result = self.store.call("query_tickets", {"operation": "top_reporter", "since": "2026-10-14T00:00:00Z"})
        self.assertEqual(result["answer"], sorted(["測試丙", "測試丁"]))

    def test_no_matches_is_not_error(self):
        result = self.store.call("query_tickets", {"operation": "count", "contains_any": ["隕石"]})
        self.assertEqual((result["answer"], result["evidence"]), (0, []))

    def test_reject_unknown_fields_types_and_tools(self):
        for tool, args in [("query_tickets", {"operation": "count", "category": "noise"}),
                           ("devices_at_location", {"location": True}), ("shell", {"command": "anything"})]:
            with self.assertRaises(ValueError):
                self.store.call(tool, args)

    def test_raw_projection_ignores_labels(self):
        rows = [{**r, "device_id": "FAKE", "category": "fake", "urgency": "high"} for r in self.fixture["tickets"]]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.assertEqual(set(self.store.tickets[0]), {"ticket_id", "created_at", "reporter", "raw_text"})
        self.assertEqual(self.store.call("query_tickets", {"operation": "count", "device_id": "D01"})["answer"], 3)

    def test_device_ids_are_not_substrings_of_other_ascii_codes(self):
        row = {**self.fixture["tickets"][0], "raw_text": "ND010 不是設備 D01 的回報"}
        self.path.write_text(json.dumps({**row, "raw_text": "ND010 面板故障"}) + "\n", encoding="utf-8")
        self.assertEqual(self.store.call("query_tickets", {"operation": "count", "device_id": "D01"})["answer"], 0)

    def test_read_evidence_matches_the_returned_records(self):
        self.path.write_text("".join(json.dumps({**self.fixture["tickets"][0], "ticket_id": f"S{i}"}) + "\n" for i in range(31)), encoding="utf-8")
        result = self.store.call("query_tickets", {"operation": "read"})
        self.assertEqual(result["evidence"], result["answer"])
        self.assertEqual(result["evidence"], [r["ticket_id"] for r in result["records"]])

    def test_dev_guard_before_file_open(self):
        with self.assertRaises(ValueError):
            Store(Path(self.temp.name) / "dev.jsonl")

    def test_duplicate_ids_fail(self):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.fixture["tickets"][0]) + "\n")
        with self.assertRaises(ValueError):
            _ = self.store.tickets

    def test_model_must_select_finish(self):
        actions = [{"tool": "query_tickets", "arguments": {"operation": "count"}},
                   {"tool": "finish", "result_id": 0}]
        result = self.run_agent(actions)
        self.assertEqual(result["answer"], len(self.fixture["tickets"]))
        self.assertEqual(result["tools_used"], ["query_tickets"])

    def test_model_can_repair_bad_arguments(self):
        actions = [{"tool": "get_device", "arguments": {"identifier": True}},
                   {"tool": "get_device", "arguments": {"identifier": "D01", "field": "device_id"}},
                   {"tool": "finish", "result_id": 1}]
        self.assertEqual(self.run_agent(actions)["answer"], "D01")

    def test_preview_never_truncates_the_final_answer(self):
        self.path.write_text("".join(json.dumps({**self.fixture["tickets"][0], "ticket_id": f"S{i}"}) + "\n" for i in range(31)), encoding="utf-8")
        result = self.run_agent([{"tool": "query_tickets", "arguments": {"operation": "ticket_ids"}},
                                 {"tool": "finish", "result_id": 0}])
        self.assertEqual(len(result["answer"]), 31)
        self.assertEqual(len(result["evidence"]), 20)

    def test_fabricated_result_rejected(self):
        with self.assertRaises(ValueError):
            self.run_agent([{"tool": "finish", "result_id": 0}])

    def test_repeat_does_not_execute_again(self):
        actions = [{"tool": "query_tickets", "arguments": {"operation": "count"}}] * 2
        actions += [{"tool": "finish", "result_id": 0}]
        self.assertEqual(self.run_agent(actions)["tools_used"], ["query_tickets"])

    def test_steps_fail_without_invented_answer(self):
        with self.assertRaises(RuntimeError):
            self.run_agent([{"tool": "query_tickets", "arguments": {"operation": "count"}}] * 2, max_steps=2)

    def test_deadline_before_model_call(self):
        with self.assertRaises(TimeoutError):
            Agent(self.store, ScriptedModel([]), time.monotonic() - 1).ask("測試")

    def test_evidence_details_fail_for_unknown_id(self):
        with self.assertRaises(ValueError):
            self.store.call("get_ticket_details", {"ticket_ids": ["NOT_REAL"]})

    def test_patrol_excludes_future_and_keeps_safety_excerpt(self):
        future = {**self.fixture["tickets"][0], "ticket_id": "FUTURE", "created_at": "2030-01-01T00:00:00Z"}
        long = {**self.fixture["tickets"][0], "ticket_id": "LONG", "raw_text": "D01 " + "前言" * 150 + "有人受困需要處理" + "後文" * 100 + "冒煙"}
        self.path.write_text("".join(json.dumps(r) + "\n" for r in [future, long]), encoding="utf-8")
        candidates = shortlist(self.store)
        self.assertEqual(len(candidates), 1)
        self.assertIn("冒煙", candidates[0]["raw_text"])
        self.assertNotIn("有人受困", candidates[0]["raw_text"])
        self.assertEqual(candidates[0]["safety_term_count"], 2)
        self.assertLessEqual(len(candidates[0]["raw_text"]), 90)

    def test_patrol_critical_gate_by_id_name_model_without_safety_words(self):
        self.store.devices = [
            {"device_id": "D01", "name": "合成主機", "model": "FAKE-01", "critical": True},
            {"device_id": "D02", "name": "合成副機", "model": "FAKE-02", "critical": False}]
        texts = ["D01 異音", "合成主機需要檢查", "FAKE-01 不運轉", "D02 起火冒煙", "D010 故障", "未指定設備冒煙"]
        rows = [{**self.fixture["tickets"][0], "ticket_id": f"FAKE{i}", "raw_text": text}
                for i, text in enumerate(texts)]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        candidates = shortlist(self.store)
        self.assertEqual([row["ticket_id"] for row in candidates], ["FAKE0", "FAKE1", "FAKE2"])
        self.assertTrue(all(row["critical_devices"][0]["device_id"] == "D01" for row in candidates))
        self.store.devices[0]["critical"] = False
        self.assertEqual(shortlist(self.store), [])

    def test_patrol_does_not_cut_off_critical_candidates_at_30(self):
        rows = [{**self.fixture["tickets"][0], "ticket_id": f"FAKE{i}", "raw_text": "D01 異音"}
                for i in range(61)]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        candidates = shortlist(self.store)
        self.assertEqual(len(candidates), 61)
        seen = []
        class Model:
            def decide(self, messages, schema, max_tokens):
                batch = json.loads(messages[1]["content"])
                seen.append([row["ticket_id"] for row in batch])
                # Synthetic deterministic preference; not a ranking accuracy test.
                ordered = sorted(batch, key=lambda row: int(row["ticket_id"][4:]), reverse=True)
                return {"items": [{"ticket_id": row["ticket_id"], "priority": "medium"} for row in ordered[:10]]}
        candidates[-1]["safety_term_count"] = 2
        candidates[-2]["safety_term_count"] = 1
        candidates[-2]["created_at"] = "2026-10-02T00:00:00Z"
        candidates[-3]["safety_term_count"] = 1
        selected = prioritize_candidates(candidates)
        self.assertEqual(len(selected), 30)
        self.assertEqual([row["ticket_id"] for row in selected[:3]], ["FAKE60", "FAKE59", "FAKE58"])
        output = rank_candidates(Model(), selected)
        self.assertEqual(len(seen), 1)
        self.assertEqual(set(seen[0]), {row["ticket_id"] for row in selected})
        self.assertEqual(output[0]["ticket_id"], "FAKE60")
        with self.assertRaises(ValueError):
            rank_candidates(Model(), candidates)
        self.assertEqual(rank_candidates(Model(), []), [])
        self.assertEqual(len(rank_candidates(Model(), candidates[:3])), 3)

    def test_patrol_safety_tickets_bypass_model_and_signature_is_dropped(self):
        self.store.devices = [{"device_id": "D01", "name": "合成主機", "model": "FAKE-01", "critical": True}]
        texts = ["D01 漏電 摸外殼會麻麻的", "D01 異音\n\n-- \n從我的 iPhone 傳送"] + [f"D01 數值亂跳{i}" for i in range(12)]
        rows = [{**self.fixture["tickets"][0], "ticket_id": f"FAKE{i}", "raw_text": text}
                for i, text in enumerate(texts)]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        candidates = prioritize_candidates(shortlist(self.store))
        self.assertEqual(candidates[0]["safety_terms"], ["漏電"])
        self.assertNotIn("iPhone", next(r for r in candidates if r["ticket_id"] == "FAKE1")["raw_text"])
        seen = []
        class Model:
            def decide(self, messages, schema, max_tokens):
                batch = json.loads(messages[1]["content"])
                seen.extend(row["ticket_id"] for row in batch)
                return {"items": [{"ticket_id": row["ticket_id"], "priority": "medium"} for row in batch[:9]]}
        output = rank_candidates(Model(), candidates)
        self.assertEqual(len(output), 10)
        self.assertEqual((output[0]["ticket_id"], output[0]["priority"]), ("FAKE0", "high"))
        self.assertIn("漏電", output[0]["reason"])
        self.assertNotIn("FAKE0", seen)
        raw = {row["ticket_id"]: row["raw_text"] for row in rows}
        for item in output:
            quote = item["reason"][item["reason"].index("「") + 1:item["reason"].index("」")]
            self.assertIn(quote, raw[item["ticket_id"]])
            self.assertNotIn("iPhone", quote)
        # Ten or more safety tickets: no model call at all.
        many = [{**row, "safety_terms": ["冒煙"], "safety_term_count": 1} for row in candidates[:10]]
        self.assertEqual([r["priority"] for r in rank_candidates(None, many)], ["high"] * 10)

    def test_patrol_rejects_duplicate_ids_and_fabricated_quotes(self):
        candidates = [{"ticket_id": "S1", "raw_text": "設備冒煙需要處理"},
                      {"ticket_id": "S2", "raw_text": "有人受困需要救援"}]
        item = {"ticket_id": "S1", "priority": "high"}
        with self.assertRaises(ValueError):
            selection_output({"items": [item, item]}, candidates)
        with self.assertRaises(ValueError):
            selection_output({"items": [item, {"ticket_id": "S2", "quote": "工單已經結案了", "priority": "low"}]}, candidates)
        output = selection_output({"items": [item, {"ticket_id": "S2", "priority": "high"}]}, candidates)
        self.assertIn("有人受困需要救援", output[1]["reason"])

    @unittest.skipUnless(os.name == "nt", "Windows Job Object lifecycle")
    def test_cli_lifetime_guard_terminates_model_child(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        guard = None
        try:
            guard = ChildGuard(process)
            guard.close()
            process.wait(timeout=3)
            self.assertIsNotNone(process.returncode)
        finally:
            if guard:
                guard.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
