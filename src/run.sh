#!/bin/bash
set -e

export HPCC_PATH=${HPCC_PATH:-/opt/hpcc}
export CUCC_PATH=${CUCC_PATH:-$HPCC_PATH/tools/cu-bridge}
export CUDA_PATH=${CUDA_PATH:-$HOME/cu-bridge/CUDA_DIR}
export CUCC_TARGETS=${CUCC_TARGETS:-htcore1000}

export PATH=$HPCC_PATH/tools/cu-bridge/tools:$HPCC_PATH/htgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=$HPCC_PATH/lib:$HPCC_PATH/htgpu_llvm/lib:/opt/htdriver/lib:$LD_LIBRARY_PATH
export CC=gcc CXX=g++

make_hpcc MAIN
