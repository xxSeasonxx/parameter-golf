#!/usr/bin/env bash
# Retired runner. Deep supervision was tested on H100 and discarded.
#
# See:
# - NEXT_EXPERIMENT.md
# - .lab/research_notes_2026-04-30.md

set -euo pipefail

echo "h100_next_deep_supervision.sh is retired: deep supervision regressed on H100."
echo "Use ./h100_ttt_eval_only.sh for the next RunPod step."
exit 1
