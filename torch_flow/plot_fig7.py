import os
import argparse
import pandas as pd
import matplotlib
from matplotlib import pyplot as plot

plot.rcParams.update({'font.size': 20})
#GEMM,activation,logit_computation,nomralization,arithmetic,pooling,interpolation,embedding,memory,roi,other
color_scheme = {"GEMM":'#4C443C' , "NonGEMM":'#DEB841', "nomralization":"#DEB841", "activation":"#769FB6", "arithmetic":"#D16666", "interpolation":"#999AC6", "memory":"#55917F",  "other":"#32373B", "pooling":"#BDBBB6", "embedding":"#83D628", "logit_computation":"#254E70", "roi":"#FAE8EB", }


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

lm = [
    #'llama2-awq',
    
    'llama2',
    'gpt2-xl',
    'gpt2-large',
    'gpt2',
    'bert',
    'mistral_MoE',
     
    # 'bert_large', 
    # 'llama3',

]

classfication = [
    'swin-base',
    # 'swin-small',
    'swin-tiny',
    # 'vit-huge',
    # 'vit-large',
    # 'vit-base',
    # 'vit-hf-base', 
    # 'vit-hf-huge', 
]

detection = [
    'detr',
    # #'maskrcnn',
    # #'fasterrcnn',
]

segmentaion = [
    #'maskformer-base',
    'segformer',
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
seq_len = {
    #'llama2-awq':2048,
    
    'llama2':2048,
    'gpt2-xl':512,
    'gpt2-large':256,
    'gpt2':256,
    'bert':128,#64,
    'bert_large':128, 
    # 'llama3':1024,
    'mistral_MoE':2048,
    }

devices = ['cpu', 'cuda']

gemm_file = "gemm.csv"
non_gemm_file = "non_gemm.csv"

def get_directories(path: str= "./non-gemm-out"):
    entries = os.listdir(path)
    # Filter only directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    return directories

def extract_non_gemm (prof_dir:str = "./non-gemm-out",out_dir = None):
    unique_non_gemm_ops = []
    direct = get_directories()
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
            if not (i in unique_non_gemm_ops):
                print (i)
                unique_non_gemm_ops.append(i)
            if (i == "aten::einsum"):
                print (f"Move this to GEMM: {dir}")

    print (unique_non_gemm_ops)

def summarize_ops (prof_dir:str = "./non-gemm-out",out_dir = None): 
    direct = get_directories(prof_dir)
    for dir in direct:
        data_file = f"{prof_dir}/{dir}/{non_gemm_file}"
        if not (os.path.exists(data_file)):
            continue
        df_nongemm = pd.read_csv(data_file)
        df_summary = pd.read_csv(f"{prof_dir}/{dir}/{dir}.csv")
        df_gng = df_summary[df_summary['name'].isin(['GEMM', 'NonGEMM'])]
        df_ops = pd.DataFrame()
        for group, list_ in ops_dict.items():
            df_ = filter_dataframes(df_summary, list_)
            df_, summary_row = sum_df_append(df_, group)
            df_ops = pd.concat([df_ops, summary_row], ignore_index=True).drop(columns = ['Unnamed: 0'])
            #print (df_)
            #break
            df_.to_csv(f"{prof_dir}/{dir}/{group}.csv")
        #break
        df_ops.to_csv(f"{prof_dir}/{dir}/summary_{dir}.csv")
        #df_gng.to_csv(f"{prof_dir}/{dir}/gng_{dir}.csv")

        df_summary_transpose = df_ops[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']
        df_gng_transpose = df_gng[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']

        df_ = df_summary_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/pct_{dir}.csv')

        df_ = df_gng_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/gng_pct_{dir}.csv')

def check_new_non_gemm (unique_non_gemm): 
    new_non_gemm = []

    for op in unique_non_gemm: 
        if not (op in non_gemm_ops): 
            new_non_gemm.append(op)
    print (f"New Non-GEMM Operators:") 
    print (new_non_gemm)

def summarize_non_gemm(prof_dir:str = "./non-gemm-out",out_dir = None):
    direct = get_directories(prof_dir)
    for dir in direct:
        data_file = f"{prof_dir}/{dir}/{non_gemm_file}"
        if not (os.path.exists(data_file)):
            continue
        df_nongemm = pd.read_csv(data_file)
        unique_nongemm = df_nongemm['name'].unique().tolist()
        check_new_non_gemm(unique_nongemm)
        df_summary = pd.read_csv(f"{prof_dir}/{dir}/{dir}.csv")
        df_gng = df_summary[df_summary['name'].isin(['GEMM', 'NonGEMM'])]
        df_summary = df_summary[df_summary['name'] == 'GEMM']
        for group, list_ in non_gemm_ops_dict.items():
            df_ = filter_dataframes(df_nongemm, list_)
            df_, summary_row = sum_df_append(df_, group)
            df_summary = pd.concat([df_summary, summary_row], ignore_index=True).drop(columns = ['Unnamed: 0'])
            #print (df_)
            #break
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
        
def plot_gng_batch(prof_dir: str = "./non-gemm-out", out_dir = ".", model_name: str ='bert', batch_size: int = 1, batch_sizes = batch_sizes, seq_len = None ):
    seq_len_ = f"_{seq_len}" if (seq_len is not None) else ""
    df_ = pd.DataFrame()
    for n in batch_sizes: 
        filename_b1 = f"{prof_dir}/{model_name}_cuda_{n}{seq_len_}/gng_pct_{model_name}_cuda_{n}{seq_len_}.csv"
    
    # filename_b2 = f"{prof_dir}/{model_name}_cuda_{2}{seq_len_}/gng_pct_{model_name}_cuda_{2}{seq_len_}.csv"
    # filename_b4 = f"{prof_dir}/{model_name}_cuda_{4}{seq_len_}/gng_pct_{model_name}_cpu_{4}{seq_len_}.csv"
    # filename_b8 = f"{prof_dir}/{model_name}_cuda_{8}{seq_len_}/gng_pct_{model_name}_cuda_{8}{seq_len_}.csv"


        if not (os.path.exists(filename_b1)): 
            print (f"We need to get CPU and/or GPU data for {model_name}_{n}{seq_len_}")
            print ("Not Generating Plots")
            return
        df_cuda = pd.read_csv(filename_b1).set_index('name')
        model_ = f"{model_name}_{n}"
        df_cuda = df_cuda.rename(columns={"pct":model_ })
        df_cuda = df_cuda [[model_]].T #df_cuda [['pct']].T
        df_ = pd.concat([df_, df_cuda])
        #print (df_)
    
    plt_ = df_.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.8, color = color_scheme)
    plt_.tick_params(labelbottom=True)
    plt_.tick_params(labelleft=True)
    out_dir = f"{out_dir}/fig7_pytorch/{model_name}_cuda/"
    os.system(f"mkdir -p {out_dir}")
    plot.savefig(f"{out_dir}/fig_7_gng_pct_{model_name}.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    df_.to_csv(f"{out_dir}/gng_pct_{model_name}_{batch_size}{seq_len_}.csv")
    print (df_cuda)  

def plot_all_gng_batch(prof_dir: str = "./non-gemm-out", out_dir = ".", batch_sizes_ = batch_sizes) : 
    models = ['swin-base', 'detr', 'swin-tiny', 'segformer', 'segformer-b1', 'segformer-b3']
    for model in models: 
        plot_gng_batch(prof_dir, out_dir, model, batch_sizes = batch_sizes_)

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
    new_column_order = ["GEMM"] + sorted_columns
    
    return new_column_order 


##########


def parse_arguments(): 
    parser = argparse.ArgumentParser(description ='Generating Figures')
    
    parser.add_argument ("--prof_dir", dest="prof_dir",
                        required = False,  type = str, help = "Directory containing profiling output", default="./non-gemm-out-test")
    
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

    summarize_non_gemm(prof_dir = prof_directory,)
    plot_all_gng_batch(prof_dir=prof_directory, out_dir=summary_directory, batch_sizes_=[1,2,4,8])
if __name__ == "__main__": 
    main()