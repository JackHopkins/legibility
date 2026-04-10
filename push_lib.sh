#!/usr/bin/env bash
# Quick commit and push for lib/ changes.
# Commit message is the current time: HH:MM:DD-MM
set -e

cd "$(dirname "$0")"

MSG=$(date +"%H:%M:%d-%m")

git add lib/
git commit -m "$MSG"
git push origin main
