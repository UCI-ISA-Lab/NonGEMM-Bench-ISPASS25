#!/bin/bash

# Set Path to TensorRT/samples/
export TRT_SAMPLES=/path/to/trt/TensorRT/samples
# Set Path to TensorRT build output 
export TRT_BUILD_OUT=/path/to/trt/TensorRT/build/out 
# Set Path to output directory containing the summarized ou
export SUMMARY_PATH=./non-gemm-summary-trt

# Set Path to the TensorRT toolchain lib
export LD_LIBRARY_PATH=/path/to/trt/TensorRT-10.4.0.26/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH

# Set Path to directory containing ONNX Models
export ONNX_DIR=/path/to/NonGEMM-Bench/onnx_flow/onnx
#
export PROF_RUNS=100
