import os
import pandas as pd
import argparse
from matplotlib import pyplot as plot

plot.rcParams.update({'font.size': 20})
#GEMM,activation,logit_computation,nomralization,arithmetic,pooling,interpolation,embedding,memory,roi,other
color_scheme = {"GEMM":'#4C443C' , "NonGEMM":'#DEB841', "nomralization":"#DEB841", "activation":"#769FB6", "arithmetic":"#D16666", "interpolation":"#999AC6", "memory":"#55917F",  "other":"#32373B", "pooling":"#BDBBB6", "embedding":"#83D628", "logit_computation":"#254E70", "roi":"#FAE8EB", "gemm latency (%)":'#4C443C' , "non_gemm latency (%)":'#DEB841', }


summary_dir = f"./non-gemm-summary-trt"

if not os.path.exists(summary_dir):
    os.system(f"mkdir -p {summary_dir}")

models = [
    'swin-b',
    'detr',
    'segformer',
    'swin-t',
]
non_gemm = ['NonGEMM']

batch_sizes = [1,2,4,8]
seq_len = {
    'llama2-awq':2048,
    'llama2':2048,
    'gpt2-xl':512,
    'gpt2-large':256,
    'gpt2':256,
    'bert':64,
    }

def plot_gng(prof_dir: str = "./non-gemm-summary-trt", out_dir = "./non-gemm-out", model_name: str ='bert', batch_size: int = 1, seq_len_ = None ): 
    
    filename = f"{prof_dir}/{model_name}/pct_summary.csv"
    # name,gemm latency (%),non_gemm latency (%),input_shape

    if not (os.path.exists(filename)): 
        print (f"We need to get TRT data for {model_name}_{batch_size}{seq_len_}")
        print ("Not Generating Plots")
        return
    df_trt = pd.read_csv(filename)
    df_trt = df_trt[['input_shape', 'gemm latency (%)', 'non_gemm latency (%)']]
    
    df_trt = df_trt.set_index('input_shape')
    print (df_trt)
    
    plt_ = df_trt.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.8, color = color_scheme)
    plt_.tick_params(labelbottom=True)
    plt_.tick_params(labelleft=True)
    plt_.set_xlabel('batch_size')
    output_dir = f"{out_dir}/{model_name}"
    os.system(f"mkdir -p {output_dir}")
    plot.savefig(f"{output_dir}/fig7_gng_pct_{model_name}.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    #print (df_cpu)

def parse_arguments(): 
    parser = argparse.ArgumentParser(description ='Generating Figures')
    
    parser.add_argument ("--prof_dir", dest="prof_dir",
                        required = False,  type = str, help = "Directory containing profiling output", default="./non-gemm-summary-trt")
    
    parser.add_argument ("--summary_dir", dest="summary_dir",
                        required = False,  type = str, help = "Directory storing the generated summaries and figures", default="summary")
    
    parser.add_argument ("--task", dest="task",
                        required = False,  type = str, help = "Task to analyze and generate figures", 
                        choices=["all", "classification", "segmentation", "detection", "lm", "quant"], default = "all")
    
    parser.add_argument ("--gng", dest="gng",
                        required = False,  action = "store_true", help = "Task to analyze and generate figures", 
                        )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    prof_directory = args.prof_dir 
    if not os.path.exists(prof_directory): 
        print(f"Make sure you have entered valid profiling outputs, {prof_directory} Not Found ")
        raise FileNotFoundError(f"Directory '{prof_directory}' does not exist")

    out_dir = './fig7_trt'
    if not os.path.exists(out_dir): 
        os.system(f"mkdir -p {out_dir}")
    
    for model in models:
        plot_gng(prof_dir= prof_directory, out_dir= out_dir , model_name= model)

if __name__ == "__main__": 
    main()