#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-aerial-vehicle-detection}"
PROJECT_PATH="${WORKSPACE}/${PROJECT_DIR}"
DATASETS_CACHE="${DATASETS_CACHE:-${WORKSPACE}/datasets}"
VENV_PATH="${VENV_PATH:-${WORKSPACE}/venv}"
LOGS_PATH="${LOGS_PATH:-${WORKSPACE}/logs}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
CODE_SOURCE="${CODE_SOURCE:-git}"
REPO_URL="${REPO_URL:-}"
GIT_REF="${GIT_REF:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
INSTALL_TORCH="${INSTALL_TORCH:-false}"
TORCH_VERSION="${TORCH_VERSION:-}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
STAGE="${1:-all}"
SECRETS_FILE="${SECRETS_FILE:-${WORKSPACE}/.train-env}"

if [ -f "${SECRETS_FILE}" ]; then
    set -a
    . "${SECRETS_FILE}"
    set +a
fi

stage_checkout() {
echo "==> Checkout stage started at $(date -u +%FT%TZ)"
nvidia-smi || true

git config --global --add safe.directory '*'

export DEBIAN_FRONTEND=noninteractive
if ! command -v rsync >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends git rsync tmux libgl1 libglib2.0-0 ca-certificates
fi

mkdir -p "${WORKSPACE}" "${DATASETS_CACHE}" "${LOGS_PATH}" "${PROJECT_PATH}"

if [ "${CODE_SOURCE}" = "git" ]; then
    if [ -z "${REPO_URL}" ] || [ -z "${GIT_REF}" ]; then
        echo "REPO_URL and GIT_REF are required for CODE_SOURCE=git" >&2
        exit 2
    fi
    cd "${PROJECT_PATH}"
    if [ ! -d .git ]; then
        git init -q
    fi
    git config --local advice.detachedHead false
    if git remote get-url origin >/dev/null 2>&1; then
        git remote set-url origin "${REPO_URL}"
    else
        git remote add origin "${REPO_URL}"
    fi

    FETCH_URL="${REPO_URL}"
    if [ -n "${GITHUB_TOKEN}" ]; then
        ASKPASS="/dev/shm/git-askpass-$$"
        printf '#!/bin/sh\necho "${GITHUB_TOKEN}"\n' > "${ASKPASS}"
        chmod 700 "${ASKPASS}"
        trap 'rm -f "${ASKPASS}"' EXIT
        export GIT_ASKPASS="${ASKPASS}"
        export GIT_TERMINAL_PROMPT=0
        FETCH_URL="$(echo "${REPO_URL}" | sed -E 's#https://#https://x-access-token@#')"
    fi

    echo "==> Fetching ${GIT_REF}"
    git fetch --depth 1 "${FETCH_URL}" "${GIT_REF}"
    git checkout -q --force FETCH_HEAD
    echo "==> Checked out $(git rev-parse --short HEAD)"
else
    cd "${PROJECT_PATH}"
    if [ ! -d .git ]; then
        git init -q
        git config --local user.email "pipeline@localhost"
        git config --local user.name "cloud-bootstrap"
        git commit -q --allow-empty -m "rsync snapshot ${GIT_REF:-unknown}"
        echo "==> WARNING: rsync mode, pipeline_version will not match the local commit"
    fi
fi

if [ ! -L "${PROJECT_PATH}/datasets" ]; then
    if [ -d "${PROJECT_PATH}/datasets" ]; then
        cp -an "${PROJECT_PATH}/datasets/." "${DATASETS_CACHE}/" 2>/dev/null || true
        rm -rf "${PROJECT_PATH}/datasets"
    fi
    ln -sfn "${DATASETS_CACHE}" "${PROJECT_PATH}/datasets"
fi
echo "==> Datasets cache: ${PROJECT_PATH}/datasets -> $(readlink -f "${PROJECT_PATH}/datasets")"
echo "==> Datasets already cached: $(du -sh "${DATASETS_CACHE}" 2>/dev/null | cut -f1)"
echo "==> Checkout stage finished"
}

stage_deps() {
echo "==> Dependency stage started at $(date -u +%FT%TZ)"
if [ ! -d "${VENV_PATH}" ]; then
    if [ "${INSTALL_TORCH}" = "true" ]; then
        echo "==> Creating isolated venv at ${VENV_PATH}"
        python3 -m venv "${VENV_PATH}"
    else
        echo "==> Creating venv at ${VENV_PATH}, inheriting torch from the image"
        python3 -m venv --system-site-packages "${VENV_PATH}"
    fi
fi
PYTHON="${VENV_PATH}/bin/python"
"${PYTHON}" -m pip install -q --upgrade pip

REQUIREMENTS_PATH="${PROJECT_PATH}/${REQUIREMENTS_FILE}"
REQUIREMENTS_STAMP="${VENV_PATH}/.requirements.sha256"
CURRENT_STAMP="$(sha256sum "${REQUIREMENTS_PATH}" | cut -d' ' -f1)"
if [ ! -f "${REQUIREMENTS_STAMP}" ] || [ "$(cat "${REQUIREMENTS_STAMP}")" != "${CURRENT_STAMP}" ]; then
    if [ "${INSTALL_TORCH}" = "true" ]; then
        echo "==> Installing torch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
        "${PYTHON}" -m pip install -q "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX_URL}"
    fi
    echo "==> Installing ${REQUIREMENTS_FILE}"
    "${PYTHON}" -m pip install -q -r "${REQUIREMENTS_PATH}"
    echo "${CURRENT_STAMP}" > "${REQUIREMENTS_STAMP}"
else
    echo "==> Dependencies already installed for this requirements file"
fi

"${PYTHON}" - <<'PYCHECK'
import numpy
import torch
print(f"numpy {numpy.__version__}")
print(f"torch {torch.__version__}, cuda available {torch.cuda.is_available()}")
torch.zeros(1).numpy()
if torch.cuda.is_available():
    print(f"device 0: {torch.cuda.get_device_name(0)}")
    print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
import ultralytics
print(f"ultralytics {ultralytics.__version__}")
PYCHECK

echo "==> Dependency stage finished at $(date -u +%FT%TZ)"
}

case "${STAGE}" in
checkout) stage_checkout ;;
deps) stage_deps ;;
all) stage_checkout; stage_deps ;;
*) echo "usage: bootstrap.sh {checkout|deps|all}" >&2; exit 2 ;;
esac
