# NonGEMM Bench ISPASS 2025 AE
This repository contains all the code to implement NonGEMM Bench, a benchmarking flow tailored for profiling non-GEMM operators in ML models. It also contains all the scripts to generate the data used in the paper Understanding the Performance Horizon of the Latest ML Workloads with NonGEMM Workloads at ISPASS 2025.

## PyTorch Flow
NonGEMM Bench PyTorch Flow leverages the PyTorch Profiler to measure the inference latency breakdown.
### Dependencies
Software:
- [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- Llama-2-7b-hf Weights
- [ImageNet Dataset](https://www.image-net.org/)
- [COCO Dataset](https://cocodataset.org/#home)
- [COCO API](https://github.com/cocodataset/cocoapi)

Hardware:
- GPU with CUDA Support (The flow was tested with CUDA 12).
### Setting Up the Environment
1. Create a new conda environment
   `cd torch_flow` 
   `conda create -n ng-torch python=3.10`
2. Install required poackages
   `pip install -r requirements.txt`
3. Install COCO API
   ```
   git clone https://github.com/cocodataset/cocoapi.git
   cd path/to/cocoapi/PythonAPI
   make install 
   ```
4. Configure `torch_flow/setup.sh` with path to datasets and to Llama weights
   i. Set the variables `COCO_IMAGES`, `COCO_ANN`, `IMAGENET_IMAGES`, and `LLAMA_WEIGHTS`.   
### Running the Scripts
```
cd path/to/NonGEMM-Bench/torch_flow
bash run_ispass_all.sh
```
The script profiles all NonGEMM Bench models, and reproduces Figures 5, 6(a), 7(Top), and 8 in the paper.\
The output data and plots are stored in `./torch_flow/summary`

## ONNX Runtime (ORT) Flow
NonGEMM Bench PyTorch Flow leverages the ORT Execution Provider Profiler to measure the inference latency breakdown.
### Setting Up the Environment 
1. Activate the `ng-torch` environment.
   `conda activate ng-torch`
2. Verify the CUDA Execution Provider (EP) is available in ORT.
   ```
   python
   import onnxruntime as ort
   ort.get_available_providers()
   ```
   i. If the CUDA EP does not appear, try adding the path to CuDNN to `LD_LIBRARY_PATH`
     `LD_LIBRARY_PATH=path/to/CuDNN/lib:$LD_LIBRARY_PATH`
### Running the Scripts
```
cd path/to/NonGEMM-Bench/torch_flow
bash run_ispass_all.sh
```
The script automatically converts the torch models to ONNX models stored in `torch_onnx/onnx`. Then it profiles all NonGEMM Bench models, and reproduces Figures 6(b) in the paper.\
The output data and plots are stored in `./onnx_flow/fig6_onnx`

## TensorRT Flow
NonGEMM Bench PyTorch Flow leverages the PyTorch Profiler to measure the inference latency breakdown.
### Dependencies
Software:
- [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT/tree/release/10.4)
- [TensorRT GA](https://developer.nvidia.com/tensorrt-getting-started)
  
Hardware:
- GPU with CUDA Support (The flow was tested with CUDA 12.6).
### Setting Up the Environment
1. Create a new conda environment
   `cd trt_flow` 
   `conda create -n ng-trt python=3.10`
2. Install required poackages
   `pip install -r requirements.txt`
3. Install and build TesnorRT GA and TensorRT OSS
   Follow the steps on the [TensorRT repository](https://github.com/NVIDIA/TensorRT/tree/release/10.4?tab=readme-ov-file) to install and build TRT OSS.
4. Configure `setup.sh` with path to TensorRT build
  Open `trt_flow/setup.sh` and set the following variables.
   ```
    # Set Path to TensorRT/samples/
    export TRT_SAMPLES=/path/to/trt/TensorRT/samples
    # Set Path to TensorRT build output 
    export TRT_BUILD_OUT=/path/to/trt/TensorRT/build/out     
    # Set Path to the TensorRT toolchain lib
    export LD_LIBRARY_PATH=/path/to/trt/TensorRT-10.4.0.26/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
       
   ```
### Running the Scripts
TensorRT requires the models to be in ONNX format. The onnx models generated in the previous steps are stored in `onnx_flow/onnx`.\
1. Configure the following variables in `setup.sh`.
   ```
    # Set Path to output directory containing the summarized outputs
    export SUMMARY_PATH=./non-gemm-summary-trt
    # Set Path to directory containing ONNX Models
    export ONNX_DIR=/path/to/NonGEMM-Bench/onnx_flow/onnx
   ```
2. Run the scripts.
   ```
    cd path/to/NonGEMM-Bench/torch_flow
    bash run_ispass_all.sh
   ```
The script reproduces Figures 7(bottom)in the paper.\
The output data and plots are stored in `./trt_flow/fig7_trt`

