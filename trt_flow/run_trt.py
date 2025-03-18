import argparse 
import subprocess
import os 

TRT_BUILD_OUT = os.getenv("TRT_BUILD_OUT")

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process the model using TensorRT')

    # Adding the required arguments
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--model_onnx_path', type=str, help='Path to the ONNX model file')
    parser.add_argument('--input_tensors', type=str, help='Input name and shape')

    # Adding the optional arguments
    parser.add_argument('--input_path', type=str, default=None, 
                        help='Path to the input data')
    parser.add_argument('--fp_16', action='store_true', 
                        help='Enable FP16 precision (optional)')
    parser.add_argument('--int8', action='store_true', 
                        help='Enable INT8 precision (optional)')
    parser.add_argument('--out_path', type=str, default='.', 
                        help='Output path for results TRT logs (default: current directory)')
    parser.add_argument('--num_prof_runs', type=int, default=5000, 
                        help='num of profiling runs')

    # Parse arguments
    args = parser.parse_args()
    fp16 = "--fp16" if args.fp_16 else " "
    out_path = f"{args.out_path}"

    os.system(f"mkdir -p {out_path}")

    trt_cmd = f"{TRT_BUILD_OUT}/trtexec --onnx={args.model_onnx_path} --shapes={args.input_tensors} {fp16} --saveEngine=./{out_path}/g1.trt --dumpLayerInfo --dumpProfile  --exportProfile={out_path}/profile.json --exportLayerInfo={out_path}/layer.json --separateProfileRun --exportTimes={out_path}/times.json --verbose --profilingVerbosity=detailed --iterations={args.num_prof_runs} --useCudaGraph > {out_path}/log.txt"
    
    x = os.system(trt_cmd)
    
    print(x)

if __name__ == "__main__": 
    main()
