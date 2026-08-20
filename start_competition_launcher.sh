#!/usr/bin/env bash
# 始终使用 Python 3，避免系统或 IDE 默认选择 Python 2。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec /usr/bin/python3 competition_launcher.py
