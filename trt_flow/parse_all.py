import os 

models = [
    "detr", 
    #"gpt2-xl", 
    #"llama2",
    "segformer", 
    "swin-b", 
    "swin-t",

]

batch_sizes = {
    "detr":[1,2,4,8], 
    # "gpt2-xl":[1],
    # "llama2":[1], 
    "segformer":[1,2,4,8],
    "swin-b":[1,2,4,8],
    "swin-t":[1,2,4,8],




}

seq_lengths = {
    "detr":"", 
    # "gpt2-xl":[512, 1024, 2048, 4096],
    # "llama2":[1024, 2048, 8192], 
    "segformer":"",
    "swin-b":"",
    "swin-t":"",


}

logs_path = "./logs"
onnx_path = os.environ.get("ONNX_DIR") 
summary_path =  os.environ.get("SUMMARY_PATH")
cmds = []
for model in models: 
    model_onnx = "swin-b.onnx" if model == "swin-b" else f"{model}_onnx/model.onnx"
    model_onnx = "swin-t.onnx" if model == "swin-t" else f"{model}_onnx/model.onnx"

    for n in batch_sizes[model]:
        if seq_lengths[model] == "":
            trt_log_path = f"{logs_path}/{model}_trt_{n}"
            if not (os.path.exists(trt_log_path)): 
                os.system(f"mkdir -p {trt_log_path}")
            cmd = f"python parse_trt.py --path_to_trt_logs {trt_log_path} --onnx_file {onnx_path}/{model_onnx} > {summary_path}/{model}/fusion_rate_{model}_{n}.txt"
            cmds.append(cmd)
            x = os.system(cmd)
            print (f"{model}_trt_{n} Returned with Code: {x}")
            
        else: 
            for s in seq_lengths[model]:
                trt_log_path = f"{logs_path}/{model}_trt_{n}_{s}"
                if not (os.path.exists(trt_log_path)): 
                    os.system(f"mkdir -p {trt_log_path}")
                cmd = f"python parse_trt.py --path_to_trt_logs {trt_log_path} --onnx_file {onnx_path}/{model_onnx} > {summary_path}/{model}/fusion_rate_{model}_s{n}_{s}.txt "
                cmds.append(cmd)
                x = os.system(cmd)
                print (f"{model}_trt_{n}_{s} Returned with Code: {x}")



