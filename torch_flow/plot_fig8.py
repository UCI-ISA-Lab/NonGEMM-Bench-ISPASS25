import os
import argparse
import pandas as pd
import matplotlib
from matplotlib import pyplot as plot


plot.rcParams.update({'font.size': 20})
#GEMM,activation,logit_computation,nomralization,arithmetic,pooling,interpolation,embedding,memory,roi,other
color_scheme = {"GEMM":'#4C443C' , "NonGEMM":'#DEB841', "nomralization":"#DEB841", "activation":"#769FB6", "arithmetic":"#D16666", "interpolation":"#999AC6", "memory":"#55917F",  "other":"#32373B", "pooling":"#BDBBB6", "embedding":"#83D628", "logit_computation":"#254E70", "roi":"#FAE8EB", "qdq":"#BDBBB6"}


summary_dir = f"summary_default"#f"./camera_summary"#f"./rebuttal_summary"
if not os.path.exists(summary_dir):
    os.system(f"mkdir -p {summary_dir}")

gemm_ops = ["aten::mm", "aten::matmul", "aten::bmm", "aten::linear", "aten::addmm", "aten::addbmm", "aten::baddbmm", "aten::mv",    "aten::dot",
    "aten::ger", "aten::matmul.out", "aten::scaled_dot_product_attention",
    "aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::conv_tbc",
    "aten::conv_transpose1d", "aten::conv_transpose2d", "aten::conv_transpose3d",
    "aten::slow_conv3d", "aten::slow_conv_dilated2d", "aten::slow_conv_dilated3d", "aten::slow_conv_transpose2d", "aten::slow_conv_transpose3d",
    "aten::thnn_conv2d","aten::thnn_conv_depthwise2d","aten::scaled_dot_product_attention", "aten::linear",'wqlinearmmfunction',"conv1d", "aten::einsum"]

attention_ops = [i for i in gemm_ops if "attention" in i]

gemm_ops_no_attn = [i for i in gemm_ops if not (i in attention_ops)]

models = [
    'swin-base',
    'swin-small',
    'swin-tiny',
    'vit-huge',
    'vit-large',
    'vit-base',
    'detr',
    'maskformer-base',
    'segformer',
    'segformer-b1',
    'segformer-b3',
    'segformer-b5',


    
    'llama2',
    'gpt2-xl',
    'gpt2-large',
    'gpt2',
    'bert',
    'maskrcnn',
    'fasterrcnn',
    'mistral_MoE',
    'llama3', 

]

quant = [
    'llama3',
    'llama3-8bit',
]

non_gemm = ['NonGEMM']
act = ['aten::silu', 'aten::gelu', 'aten::sigmoid', 'aten::relu', 'aten::relu_', 'newgeluactivation_prof', 'triton_poi_fused_mul_silu_8',  ]
logit_computation = ['aten::softmax',]
norm = ['aten::layer_norm', 'aten::group_norm', 'aten::batch_norm', 'llamarmsnorm_prof', "detrfrozenbatchnorm2d_prof", "mixtralrmsnorm_prof", "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0", 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7', 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9', 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_10',  ]

roi = ['torchvision::roi_align', 'torchvision::nms', ]
arith = [ 'aten::add', 'aten::add_', 'aten::div', 'aten::mul', 'aten::floor', 'aten::neg',  'aten::mul_', 'aten::gt', 'aten::sub','aten::ge', 'aten::lt', 'aten::le', 'aten::eq', 'aten::ne', 'aten::bitwise_not',  'aten::__and__', 'aten::is_nonzero', 'aten::clamp', 'aten::all', 'aten::pow', 'aten::sin', 'aten::cos', 'aten::rsqrt', 'aten::sqrt', 'aten::log2', 'aten::exp', 'aten::max', 'aten::min', 'aten::cumsum', "aten::mean", "aten::div_", "aten::index_add_", 'aten::__or__', "aten::argmax", 'aten::exponential_', 'aten::sum', 'aten::bitwise_and',  'triton_red_fused_add_all_eq_masked_fill_1', 'triton_poi_fused_add_cat_clone_mul_4', 'triton_poi_fused_add_all_bitwise_not_constant_pad_nd_eq_masked_fill_mul_6', 'aten::rsub', 'aten::abs',  ]

arith_lin_elmt_wise = [ 'aten::add', 'aten::add_', 'aten::div', 'aten::mul', 'aten::floor', 'aten::neg',  'aten::mul_', 'aten::gt', 'aten::sub','aten::ge', 'aten::lt', 'aten::le', 'aten::eq', 'aten::ne', 'aten::bitwise_not',  'aten::__and__', 'aten::is_nonzero', 'aten::clamp', 'aten::all','aten::split',  ]

arith_non_lin_elmt_wise = ['aten::pow', 'aten::sin', 'aten::cos', 'aten::rsqrt', 'aten::sqrt', 'aten::log2', 'aten::exp',]

arith_lin_red = ['aten::max', 'aten::min', 'aten::cumsum',   ]

pooling = ['aten::adaptive_avg_pool1d','aten::max_pool2d', 'aten::adaptive_avg_pool2d',  ]

interpolation = ['aten::upsample_nearest2d', 'aten::upsample_bilinear2d',  ]

embedding = ['aten::embedding',]

mem = ['aten::slice', 'aten::view', 'aten::permute', 'aten::transpose', 
       'aten::reshape',  'aten::flatten', 'aten::pad', 'aten::contiguous',  
       'aten::index', 'aten::unsqueeze', 'aten::to', 'aten::cat', 'aten::copy_', 
       'aten::empty', 'aten::expand', 'aten::new_empty', 'aten::new_zeros', 
       'aten::where',  'aten::unbind',  'aten::select', 'aten::new_full', 
       'aten::masked_fill', 'aten::ones', 'aten::fill_', 'aten::full', 
       'aten::repeat', 'aten::stack',  'aten::arange',  'aten::type_as', 
       'aten::_unique2', 'aten::index_put_', 'aten::zeros',   'aten::zeros_like', 
       'aten::expand_as', 'aten::full_like',  'aten::detach',   'aten::detach_', 
       'aten::split_with_sizes',"aten::one_hot", "aten::scatter", "aten::new_ones", 
       'aten::squeeze', 'aten::clone', 'aten::masked_fill_', 'aten::ones_like', 
       'aten::empty_like', 'aten::resize_' , 'triton_poi_fused__to_copy_2', 
       'triton_poi_fused__to_copy_3', 'triton_poi_fused_clone_5',  
       'triton_poi_fused__to_copy_11', 'aten::_unsafe_view', 'aten::roll',
       'aten::split', 'aten::t', 'aten::any', 'aten::argwhere',   ]

other = ['aten::dropout', 'aten::lift_fresh', 'aten::meshgrid', 'aten::topk', 'aten::sort', 'aten::_assert_async', 'aten::_has_compatible_shallow_copy_type',  ]

qdq = ["bit_8_quantize_void kint8vectorquant<__half, 1024, 1>(__half*, signed char*, float*, float, int, int)", 
       "bit_8_dequantize_void kdequant_mm_int32_fp16<4, 512>(int*, float*, float*, __half*, __half*, int, int, int)"]

non_gemm_ops = act + logit_computation + norm + roi + arith + pooling + interpolation + embedding + mem + other
non_gemm_ops_dict = {'activation':act, "logit_computation":logit_computation,
                     'nomralization':norm, 'arithmetic':arith, "pooling":pooling,
                     'interpolation':interpolation, 'embedding': embedding,
                     'memory':mem, 'roi':roi, 'other':other,'qdq':qdq}

gemm_ops_dict = {
    "gemm":gemm_ops_no_attn, 
    "attention":attention_ops,
}

ops_dict = {
    "gemm":gemm_ops_no_attn, 
    "attention":attention_ops,
    'activation':act, "logit_computation":logit_computation,
    'nomralization':norm, 'arithmetic':arith, "pooling":pooling,
    'interpolation':interpolation, 'embedding': embedding,
    'memory':mem, 'roi':roi, 'other':other,

}

batch_sizes = [1,2,4,8]#[1,8]#[1,2,4,8]#[1]#

devices = ['cpu', 'cuda']

gemm_file = "gemm.csv"
non_gemm_file = "non_gemm.csv"

def get_directories(path: str= "./non-gemm-out"):
    entries = os.listdir(path)
    # Filter only directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    return directories

#### Processing Quantization 
def extract_quantization_ops (q_bit: str = "8bit", prof_dir:str = "./non-gemm-out",out_dir = None): 
    quantization_ops = []

    direct = get_directories(prof_dir)
    print (direct)
    for dir in direct:
        data_file = f"{prof_dir}/{dir}/{non_gemm_file}"
        if not (os.path.exists(data_file)):
            continue
        df_nongemm = pd.read_csv(data_file)
        tmp = df_nongemm['name'].unique().tolist()
        tmp.remove('Inference')
        tmp.remove('NonGEMM')

        for i in tmp:
            if (q_bit in i) and not (i in quantization_ops):
                print (i)
                quantization_ops.append(i)
            if (i == "aten::einsum"):
                print (f"Move this to GEMM: {dir}")
    return quantization_ops
    print (unique_non_gemm_ops)

def summarize_non_gemm_quant(prof_dir:str = "./non-gemm-out",out_dir = None):
    direct = get_directories(prof_dir)
    for dir in direct:
        data_file = f"{prof_dir}/{dir}/{non_gemm_file}"
        if not (os.path.exists(data_file)):
            continue
        df_nongemm = pd.read_csv(data_file)
        df_summary = pd.read_csv(f"{prof_dir}/{dir}/{dir}.csv")
        df_gng = df_summary[df_summary['name'].isin(['GEMM', 'NonGEMM'])]
        df_summary = df_summary[df_summary['name'] == 'GEMM']
        df_nongemm['name']= df_nongemm['name'].str.replace('_q8bit', '', regex=False)
        for group, list_ in non_gemm_ops_dict.items():
            df_ = filter_dataframes_q(df_nongemm, list_)
            df_, summary_row = sum_df_append(df_, group)
            df_summary = pd.concat([df_summary, summary_row], ignore_index=True).drop(columns = ['Unnamed: 0'])
            
            # if (group == 'other'):
                
            #     print(df_nongemm)
                
            #     print (df_)
            #     print(list_)
            # #break
            df_.to_csv(f"{prof_dir}/{dir}/{group}.csv")
        #break
        df_summary.to_csv(f"{prof_dir}/{dir}/summary_{dir}.csv")
        df_gng.to_csv(f"{prof_dir}/{dir}/gng_{dir}.csv")

        df_summary_transpose = df_summary[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']
        df_gng_transpose = df_gng[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']

        df_ = df_summary_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/pct_{dir}.csv')

        df_ = df_gng_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/gng_pct_{dir}.csv')

def plot_quant(prof_directory: str = "./non-gemm-out", summary_dir = summary_dir):
    task = "quant"
    devices = ['cuda']
    n = 1
    seq_lens = [512, 1024, 2048, 4096, 8192]
    for device in devices:
        df_cls = None
        for s in seq_lens:
            for model in quant:

                filename = f"{prof_directory}/{model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv"
                model_ = f"{model}_{device}_{s}"
                if not os.path.exists(filename):
                    print(f"File {model}_{device}_{n}/pct_{model}_{device}_{n}_{s}.csv does not exist.")
                    print (f"Skipping...")
                    continue
                df_ = pd.read_csv(filename)
                df_ [model_] = df_['pct']
                df_t = df_.set_index('name')
                df_t = df_t [[model_]]
                #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                print (df_t.T.reset_index().columns)
                df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
        if (df_cls is not None):
            out_dir = f"{summary_dir}/fig8/"
            os.system(f"mkdir -p {out_dir}")
            df_cls.set_index('index').to_csv(f"{out_dir}/{task}_{device}_pct.csv")
    #plot_figure_op_breakdown(summary_dir, task,)

def plot_quant_latency(prof_directory: str = "./non-gemm-out", summary_dir = summary_dir):
    task = "quant"
    devices = ['cuda']
    n = 1
    seq_lens = [512, 1024, 2048, 4096, 8192]
    for device in devices:
        df_cls = None
        for s in seq_lens:
            for model in quant:

                filename = f"{prof_directory}/{model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv"
                model_ = f"{model}_{device}_{s}"
                if not os.path.exists(filename):
                    print(f"File {model}_{device}_{n}/pct_{model}_{device}_{n}_{s}.csv does not exist.")
                    print (f"Skipping...")
                    continue
                df_ = pd.read_csv(filename)
                df_ [model_] = df_['total_time (us)']
                df_t = df_.set_index('name')
                df_t = df_t [[model_]]
                #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                print (df_t.T.reset_index().columns)
                df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
        if (df_cls is not None):
            out_dir = f"{summary_dir}/fig8/"
            os.system(f"mkdir -p {out_dir}")
            df_cls.set_index('index').to_csv(f"{out_dir}/{task}_{device}_latency.csv")
    #plot_figure_op_breakdown(summary_dir, task,)

def plot_figure_op_breakdown_quant(summary_directory: str="./summary", task: str ="classification", op_order: list = []):

    cuda_file = f"{summary_directory}/fig8/{task}_cuda_pct.csv"
    cuda_df = pd.read_csv(cuda_file)
    

    cuda_order = sort_df_cols(cuda_df)
    

    cuda_df = cuda_df[cuda_order].set_index('index')
    


    #plot cuda
    plt_cuda = cuda_df.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = color_scheme)#figsize = (6,4)
    plt_cuda.tick_params(labelbottom=True)
    plt_cuda.tick_params(labelleft=True)
    out_dir = f"{summary_directory}/fig8/"
    os.system(f"mkdir -p {out_dir}")
    plot.savefig(f"{out_dir}/{task}_cuda.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    

#################

## utils##
def filter_dataframes(df, list):
    # Filter DataFrame rows where "name" is in the current list
    df_ = df[df['name'].isin(list)]
    return df_

def filter_dataframes_q(df, list):
    # Filter DataFrame rows where "name" is in the current list
    df_ = df[df['name'].replace("_q8bit","").isin(list)]
    return df_

def sum_df_append (filtered_df, name):
    summed_row = filtered_df.drop(columns=["name"]).sum()
    # Add a new row with the sum and a custom 'name' value
    summed_row["name"] = name
    df = pd.concat([filtered_df, pd.DataFrame([summed_row])], ignore_index=True)
    # summary_row = filtered_df.drop(columns=["name"], errors='ignore').sum(numeric_only=True)
    # summary_row["name"] = name  # Add the list's name
    # filtered_df = pd.concat([filtered_df, summary_row.to_frame()])
    return df, summed_row.to_frame().T

def get_percentages(df):
    df_ = df.drop(columns=["Unnamed: 0"])

def sort_df_cols(df): 
    sorted_list = []
    columns = list(df.columns)
    columns.remove("GEMM")
    columns.remove("index")

    df_ = df[columns]
    df_ = df_.loc[0, :].to_dict()
    sorted_keys = dict(sorted(df_.items(), key=lambda k: k[1], reverse = True))
    sorted_columns = list(sorted_keys.keys())
    new_column_order = ["index"]+["GEMM"] + sorted_columns
    
    return new_column_order 


##########

def parse_arguments(): 
    parser = argparse.ArgumentParser(description ='Generating Figures')
    
    parser.add_argument ("--prof_dir", dest="prof_dir",
                        required = False,  type = str, help = "Directory containing profiling output", default="./non-gemm-out-quantization-server")
    
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
        print("Make sure you have entered valid profiling outputs")
        raise FileNotFoundError(f"Directory '{prof_directory}' does not exist")

    summary_directory = args.summary_dir 
    if not os.path.exists(summary_directory): 
        os.system(f"mkdir -p {summary_directory}")
    task = args.task
    gng = args.gng

    q = extract_quantization_ops(prof_dir=prof_directory)
    summarize_non_gemm_quant(prof_dir=prof_directory,)
    plot_quant(prof_directory=prof_directory,summary_dir=summary_directory)
    plot_quant_latency(prof_directory=prof_directory, summary_dir= summary_directory) 
    plot_figure_op_breakdown_quant(summary_directory=summary_directory,task = "quant",)

if __name__ == "__main__": 
    main()