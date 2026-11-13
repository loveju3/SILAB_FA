#!/bin/zsh
cd -- "${0:A:h}" || exit 1
export PYTHONDONTWRITEBYTECODE=1
if ! command -v python3 >/dev/null 2>&1; then
  echo "找不到 Python 3，請先安裝 Python 3。"
  read -r "?按 Enter 關閉視窗"
  exit 1
fi
python3 ui_server.py
if [[ $? -ne 0 ]]; then
  read -r "?啟動失敗，請查看上方訊息。按 Enter 關閉視窗"
fi
