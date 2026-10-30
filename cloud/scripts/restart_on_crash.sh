#!/usr/bin/env bash
set -uo pipefail

MAX_RESTARTS="${1:?usage: restart_on_crash.sh <max_restarts> <resume_arguments> <command>}"
RESUME_ARGUMENTS="${2:?usage: restart_on_crash.sh <max_restarts> <resume_arguments> <command>}"
COMMAND="${3:?usage: restart_on_crash.sh <max_restarts> <resume_arguments> <command>}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-15}"

CRASH_SIGNAL_EXIT_CODES=" 134 135 137 139 "

attempt=0
bash -c "${COMMAND}"
code=$?

while [[ "${CRASH_SIGNAL_EXIT_CODES}" == *" ${code} "* ]] && [ "${attempt}" -lt "${MAX_RESTARTS}" ]; do
    attempt=$((attempt + 1))
    echo "Task process was killed (exit ${code}). Restart ${attempt}/${MAX_RESTARTS} from the last checkpoint in ${RESTART_DELAY_SECONDS}s" >&2
    sleep "${RESTART_DELAY_SECONDS}"
    bash -c "${COMMAND} ${RESUME_ARGUMENTS}"
    code=$?
done

exit "${code}"
