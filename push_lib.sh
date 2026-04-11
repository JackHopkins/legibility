#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
MSG=$(date +"%H:%M:%d-%m")
git add lib/
git commit -m "$MSG"
git push origin main
