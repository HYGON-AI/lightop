#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/hyhal/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/hyhal/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
