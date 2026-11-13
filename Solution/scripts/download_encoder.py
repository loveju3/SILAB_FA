#!/usr/bin/env python3
"""Explicit online setup only; runtime encoder loading is always local/offline."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT = Path(__file__).resolve().parents[1] / "models/bge-small-zh-v1.5"


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--revision", default=REVISION, help="固定模型 commit，預設使用本次實測版本")
    args = parser.parse_args()
    revision = args.revision
    directory = args.output_dir.resolve()
    snapshot_download(REPO, revision=revision, local_dir=directory,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"])
    required = ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt")
    if any(not (directory / name).is_file() for name in required):
        raise RuntimeError("encoder 檔案不完整，請檢查下載結果")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in directory.iterdir() if p.is_file() and p.name != "encoder_manifest.json"}
    manifest = {"repo": REPO, "revision": revision, "files_sha256": files,
                "bytes": sum((directory / name).stat().st_size for name in files), "license": "MIT"}
    (directory / "encoder_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(directory), "revision": revision, "bytes": manifest["bytes"]}))


if __name__ == "__main__":
    main()
