import os 

models = [
    "detr",
    "segformer", 
    "swin-b", 
    "swin-t",
]

batch_sizes = {
    "detr":[1,2,4,8], 
    "segformer":[1,2,4,8],
    "swin-b":[1,2,4,8],
    "swin-t":[1,2,4,8],

}

seq_lengths = {
    "detr":"", 
    # "gpt2-xl":[512, 1024, 2048, 4096],
    # "llama2":[1024, 2048, 4096, 8192], 
    "segformer":"",
    "swin-b":"",
    "swin-b":"",

}

logs_path = "./logs"
summary_path =  os.environ.get("SUMMARY_PATH")
if not (os.path.exists(summary_path)): 
    os.system(f"mkdir -p {summary_path}")
for model in models: 
    for n in batch_sizes[model]:
        if seq_lengths[model] == "":
            trt_log_path = f"{logs_path}/{model}_trt_{n}"
            if not (os.path.exists(trt_log_path)): 
                os.system(f"mkdir -p {trt_log_path}")
            cmd = f"python analyze_trt.py --path_to_trt_logs {trt_log_path} --out_dir {summary_path}/{model} --model_name {model} --input_shape {n} > {summary_path}/summary_{model}_{n}.txt"
            x = os.system(cmd)
            print (f"{model}_trt_{n} Returned with Code: {x}")
        else: 
            for s in seq_lengths[model]:
                trt_log_path = f"{logs_path}/{model}_trt_{n}_{s}"
                if not (os.path.exists(trt_log_path)): 
                    os.system(f"mkdir -p {trt_log_path}")
                cmd = f"python analyze_trt.py --path_to_trt_logs {trt_log_path} --out_dir {summary_path}/{model} --model_name {model} --input_shape {n},{s} > {summary_path}/summary_{model}_{n}_{s}.txt"
                x = os.system(cmd)
                print (f"{model}_trt_{n}_{s} Returned with Code: {x}")
