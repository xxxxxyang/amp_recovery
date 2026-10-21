#!/usr/bin/env bash
set -euo pipefail

version="1.22.0"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
thirdparty_dir="${repo_root}/deploy/thirdparty"

case "$(uname -m)" in
  x86_64) ort_arch="x64" ;;
  aarch64) ort_arch="aarch64" ;;
  *)
    printf 'Unsupported CPU architecture: %s\n' "$(uname -m)" >&2
    exit 1
    ;;
esac

bundle_name="onnxruntime-linux-${ort_arch}-${version}"
bundle_dir="${thirdparty_dir}/${bundle_name}"
if [[ -f "${bundle_dir}/lib/libonnxruntime.so.${version}" ]]; then
  printf 'ONNX Runtime already installed: %s\n' "${bundle_dir}"
  exit 0
fi

mkdir -p "${thirdparty_dir}"
archive="$(mktemp)"
trap 'rm -f "${archive}"' EXIT
url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${bundle_name}.tgz"

printf 'Downloading ONNX Runtime %s for %s...\n' "${version}" "${ort_arch}"
if ! curl --ipv4 --http1.1 --fail --location \
  --retry 5 --retry-all-errors --retry-delay 2 --retry-max-time 90 \
  "${url}" --output "${archive}"; then
  printf 'Error: failed to download ONNX Runtime after 5 retries.\n' >&2
  printf 'Check your network, proxy, or firewall access to GitHub, then rerun this script.\n' >&2
  printf 'Download URL: %s\n' "${url}" >&2
  exit 1
fi

tar -xzf "${archive}" -C "${thirdparty_dir}"

if [[ ! -f "${bundle_dir}/lib/libonnxruntime.so.${version}" ]]; then
  printf 'ONNX Runtime archive did not contain the expected library: %s\n' "${bundle_dir}" >&2
  exit 1
fi

printf 'Installed ONNX Runtime to %s\n' "${bundle_dir}"
