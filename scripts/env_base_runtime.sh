#!/usr/bin/env bash

cuda_home=${CUDA_HOME:-/usr/local/cuda}
cc=${CC:-$(command -v gcc)}
cxx=${CXX:-$(command -v g++)}

if [[ -z "$cc" || -z "$cxx" ]]; then
  echo "gcc and g++ are required for source builds." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
  echo "CUDA toolkit with nvcc is required at $cuda_home." >&2
  return 1 2>/dev/null || exit 1
fi

export CC="$cc"
export CXX="$cxx"
export CUDAHOSTCXX="$cxx"
export CUDA_HOME="$cuda_home"
export PATH="$cuda_home/bin:$PATH"
