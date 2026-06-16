#!/usr/bin/env bash

toolset_root=/opt/rh/gcc-toolset-12/root/usr
cuda_home=${CUDA_HOME:-/usr/local/cuda-12.8}

if [[ ! -x "$toolset_root/bin/gcc" || ! -x "$toolset_root/bin/g++" ]]; then
  echo "GCC Toolset 12 is required at $toolset_root/bin." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
  echo "CUDA toolkit with nvcc is required at $cuda_home." >&2
  return 1 2>/dev/null || exit 1
fi

export CC="$toolset_root/bin/gcc"
export CXX="$toolset_root/bin/g++"
export CUDAHOSTCXX="$toolset_root/bin/g++"
export CUDA_HOME="$cuda_home"
export PATH="$toolset_root/bin:$cuda_home/bin:$PATH"
