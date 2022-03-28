#!/bin/bash
set -e

readonly WORKSPACE=$(./bazel info $@ workspace)
readonly EXEC_ROOT=$(./bazel info $@ execution_root)
readonly COMPDB_FILE="${WORKSPACE}/compile_commands.json"

./bazel build $@ --aspects=@bazel-compdb//:aspects.bzl%compilation_database_aspect --output_groups=compdb_files,header_files ...

# Merge compile commands.
rm -f ${COMPDB_FILE}

echo "[" > "${COMPDB_FILE}"
find "${EXEC_ROOT}" -name '*.compile_commands.json' -not -empty -exec bash -c 'cat "$1" && echo ,' _ {} \; \
  >> "${COMPDB_FILE}"
echo "]" >> "${COMPDB_FILE}"

sed -i.bak -e "s|__EXEC_ROOT__|${WORKSPACE}|" "${COMPDB_FILE}"  # Replace exec_root marker

# This is for libclang to help find source files from external repositories.
ln -f -s "${EXEC_ROOT}/external" "${WORKSPACE}/external"
rm ${COMPDB_FILE}.bak
