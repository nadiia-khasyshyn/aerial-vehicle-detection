#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
LOGS_PATH="${LOGS_PATH:-${WORKSPACE}/logs}"
RUN_ID="${RUN_ID:-task}"
LOG_FILE="${LOGS_PATH}/${RUN_ID}.log"
DONE_FILE="${LOGS_PATH}/${RUN_ID}.done"
PID_FILE="${LOGS_PATH}/${RUN_ID}.pid"

EXIT_TASK_LOST=254

task_alive() {
    [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

mode="${1:-}"

case "${mode}" in
start)
    shift
    mkdir -p "${LOGS_PATH}"
    rm -f "${DONE_FILE}" "${PID_FILE}"
    : > "${LOG_FILE}"

    launcher="nohup"
    if command -v setsid >/dev/null 2>&1; then
        launcher="setsid nohup"
    fi

    ${launcher} bash -lc "echo \$\$ > ${PID_FILE}; ( $1 ); echo \$? > ${DONE_FILE}" >> "${LOG_FILE}" 2>&1 &

    for _ in $(seq 1 10); do
        if [ -f "${PID_FILE}" ] || [ -f "${DONE_FILE}" ]; then
            break
        fi
        sleep 1
    done

    if [ ! -f "${PID_FILE}" ] && [ ! -f "${DONE_FILE}" ]; then
        echo "Failed to start task ${RUN_ID}, see ${LOG_FILE}" >&2
        tail -n 20 "${LOG_FILE}" >&2 || true
        exit 1
    fi

    echo "Started task ${RUN_ID}, pid $(cat "${PID_FILE}" 2>/dev/null || echo unknown), log ${LOG_FILE}"
    ;;
follow)
    while [ ! -f "${LOG_FILE}" ]; do sleep 2; done
    tail -n +1 -f "${LOG_FILE}" &
    TAIL_PID=$!
    stop_tail() { { kill "${TAIL_PID}"; wait "${TAIL_PID}"; } 2>/dev/null || true; }
    trap stop_tail EXIT

    while [ ! -f "${DONE_FILE}" ]; do
        if [ -f "${PID_FILE}" ] && ! task_alive; then
            sleep 3
            if [ ! -f "${DONE_FILE}" ]; then
                sleep 2
                stop_tail
                echo "Task ${RUN_ID} died without writing an exit code (killed or OOM), see ${LOG_FILE}" >&2
                exit "${EXIT_TASK_LOST}"
            fi
        fi
        sleep 5
    done

    sleep 3
    stop_tail
    exit "$(cat "${DONE_FILE}")"
    ;;
status)
    if [ -f "${DONE_FILE}" ]; then
        echo "finished:$(cat "${DONE_FILE}")"
    elif task_alive; then
        echo "running:$(cat "${PID_FILE}")"
    elif [ -f "${PID_FILE}" ]; then
        echo "lost:$(cat "${PID_FILE}")"
    else
        echo "unknown"
    fi
    ;;
stop)
    if [ -f "${PID_FILE}" ]; then
        pkill -TERM -P "$(cat "${PID_FILE}")" 2>/dev/null || true
        kill -TERM "$(cat "${PID_FILE}")" 2>/dev/null || true
    fi
    ;;
*)
    echo "usage: run_task.sh {start <command>|follow|status|stop}" >&2
    exit 2
    ;;
esac
