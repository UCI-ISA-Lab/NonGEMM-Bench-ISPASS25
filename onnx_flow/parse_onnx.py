import pandas as pd 
import json 
import glob 
import argparse
import os

model_run = ['model_run']


non_gemm_cum = ['ReorderInput','ReorderOutput','Mod', 'ScatterND', 'Not', 'GlobalAveragePool','ReduceMax', 'Reciprocal', 'ReduceMin', 'Min', 'Floor', 'Ceil', 'Pad', 'Range', 'Clip', 'Exp', 'TopK', 'Max', 'GreaterOrEqual', 'And', 'NonZero', 'ReduceProd', 'NonMaxSuppression', 'Sqrt', 'Log', 'RoiAlign', 'ScatterElements', 'Greater', 'ReduceMean', 'SimplifiedLayerNormalization', 'Gelu' ,'Tile', 'ConstantOfShape', 'MaxPool', 'Relu', 'Resize', 'CumSum', 'Cos', 'Sin', 'Flatten', 'Equal', 'Expand', 'Sigmoid','Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Add', 'LayerNormalization', 'Slice', 'Squeeze', 'Split', 'Transpose', 'Sub', 'Cast', 'Pow', 'Div', 'Where', 'Mul', 'Softmax', 'FastGelu','MemcpyToHost', 'MemcpyFromHost', 'BiasGelu', 'Neg', 'Less', 'QuickGelu']
gemm = ['MatMul', 'Gemm', 'FusedMatMul', 'Conv', 'FusedConv', 'FusedGemm','ConvTranspose',] # FusedMatMul: MatMul with Transpose
activation = ['QuickGelu', 'BiasGelu', 'FastGelu', 'Relu', 'Gelu' , 'Sigmoid', 'QuickGelu']
normalization = ['LayerNormalization', 'SimplifiedLayerNormalization',]
logit_computation = ['Softmax',] 
arithmetic = ['Pow', 'Div', 'Where', 'Mul','Sub','Add','CumSum', 'Cos', 'Sin','ReduceMean','Greater','Sqrt', 'Log', 'Max', 'GreaterOrEqual', 'And', 'NonZero', 'ReduceProd', 'Exp', 'ReduceMax', 'Reciprocal', 'ReduceMin', 'Min', 'Floor', 'Ceil', 'Not','Mod', 'Equal','Neg',  'TopK', 'Less', ] 
memory = ['ReorderInput','ReorderOutput','Unsqueeze', 'Concat', 'Reshape', 'Resize',  'Flatten', 'Expand', 'Shape', 'Split', 'Transpose', 'Slice', 'Squeeze', 'Cast', 'ScatterND','Pad','Range','Clip', 'ScatterElements', 'Tile', 'Gather', 'ConstantOfShape',] 
roi = ['NonMaxSuppression',   'RoiAlign',] 
pooling = ['MaxPool','GlobalAveragePool',] 
#non_deterministic = ['Loop','If', ]
comm = ['MemcpyToHost', 'MemcpyFromHost',]

def summarize_onnx_profile (df,model_name = None): 
    #df = pd.read_json(file)
    if (df.empty): 
        return df
    df = df[df['dur'] > 0]

    df_arg = df
    df_arg = df_arg[df_arg['args'] != {}]

    arg = []
    for i in range (df_arg.shape[0]): 
        entry = df_arg.iloc[i,7]
        if ('thread_scheduling_stats' in entry.keys()):
            del entry['thread_scheduling_stats']
        arg.append(entry)
    def extract_arg(dict, arg): 
        if (arg in dict.keys()): 
            return dict[arg]
        else: 
            return None 
    
    def print_arg (dict, msg): 
        print (type(dict))
        print (msg)
    to_parse = ['op_name', 'output_type_shape', 'output_size',  'parameter_size', 'activation_size', 'node_index', 'input_type_shape', 'provider']
    for i in to_parse:
        df_arg [i] = df_arg['args'].apply(extract_arg, arg = i)
    df_arg = df_arg.drop(['pid', 'tid','ts','ph','args'], axis = 1)
    df = df_arg
    ## Analyzing the data 
    ops = df['op_name'].unique().tolist()
    print(ops)
    providers = df['provider'].unique().tolist()
    df_gemm = df[df['op_name'].isin(gemm)]
    df_non_gemm = df[df['op_name'].isin(non_gemm_cum)]
    df_other = df[~df['op_name'].isin(gemm + non_gemm_cum)]
    print (f"Len Total: {len(df)}")
    print (f"Len GEMM: {len(df_gemm)}")
    print (f"Len NonGEMm: {len(df_non_gemm)}")
    print (f"Len Other: {len(df_other)}")
    
    print(f"ATTENTION")
    total = df['dur'].sum()
    total_gemm = df_gemm ['dur'].sum()
    total_non_gemm = df_non_gemm['dur'].sum()
    print (f'Total Inference time: {total} us')
    print (f'Total GEMM time: {total_gemm} us, {total_gemm / total * 100} %')
    print (f'Total non-GEMM time: {total_non_gemm} us, {total_non_gemm / total * 100} %')
    print (f"Gemm + NonGEMM == Total: {len(df_gemm) + len(df_non_gemm)} == {len(df)}")

    if (len(df_other) > 0): 
        print(f"SOS Missed Operators: {df_other['op_name'].unique().tolist()}")

    stats = {'total_inference':total, 'gemm':total_gemm, 'non_gemm':total_non_gemm}
    
    df_total_stats = pd.DataFrame({'dur':total, 'Ops':'total'}, index = [0])#, index = 'Total', 'gemm':total_gemm, 'non_gemm':total_non_gemm})
    
    df_total_stats.loc[len(df_total_stats)] = {'dur':total_gemm, 'Ops':'gemm'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':total_non_gemm, 'Ops':'non_gemm'}
    
    #grouping by operators: 
    df_activation = df_non_gemm [df_non_gemm['op_name'].isin(activation)]
    df_normalization = df_non_gemm [df_non_gemm['op_name'].isin(normalization)]
    df_logit_computation = df_non_gemm [df_non_gemm['op_name'].isin(logit_computation)]
    df_arithmetic = df_non_gemm [df_non_gemm['op_name'].isin(arithmetic)]
    df_memory = df_non_gemm [df_non_gemm['op_name'].isin(memory)]
    df_roi = df_non_gemm [df_non_gemm['op_name'].isin(roi)]
    df_pooling = df_non_gemm [df_non_gemm['op_name'].isin(pooling)]
    #df_non_deterministic = df_non_gemm [df_non_gemm['op_name'].isin(non_deterministic)]
    df_comm = df_non_gemm [df_non_gemm['op_name'].isin(comm)]
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_activation['dur'].sum(), 'Ops':'activation'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_normalization['dur'].sum(), 'Ops':'normalization'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_logit_computation['dur'].sum(), 'Ops':'logit_computation'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_memory['dur'].sum(), 'Ops':'memory'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_roi['dur'].sum(), 'Ops':'roi'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_pooling['dur'].sum(), 'Ops':'pooling'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':total - (total_gemm + total_non_gemm), 'Ops':'Other'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_comm['dur'].sum(), 'Ops':'comm'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_arithmetic['dur'].sum(), 'Ops':'arithmetic'}

    #df_total_stats = df_total_stats.rename(columns={'dur': f'{model_name}'})
    df_total_stats = df_total_stats.set_index('Ops').T
    return df_total_stats 


def remove_warmup(prof_dir,data_file, out_dir, drop_first):
    # Load the JSON file
    print("KARAMI")
    print(data_file)
    with open(f'{data_file}', 'r') as file:
        data = json.load(file)

    print (data_file)
    # Initialize the two lists for the divided entries
    file2_entries = []

    # Variable to track if we've reached the condition
    found_model_run = False
    #prefill_found = False 
    # Iterate through the entries and divide them
    i = 0 
    for entry in data:
        if (not found_model_run) and drop_first:
            if not found_model_run and entry.get("name") == "model_run":
                if (i > 1):
                    found_model_run = True
                i += 1
        else:
            file2_entries.append(entry)
    print (len(file2_entries))
    # Write the divided data to separate files
    data_file = f'{prof_dir}/tmp.json'
    # with open(data_file, 'w') as file1:
    #     json.dump(file2_entries, file1)
    # df = pd.read_json(data_file)
    df = pd.DataFrame(file2_entries)
    # decode = f'{out_dir}/Decode.json'
    # with open(f'{out_dir}/Decode.json', 'w') as file2:
    #     json.dump(file2_entries, file2)

    print("Files successfully split!")
    print (type(file2_entries))
    print(df.head())

    
    return df

def get_files(path: str= "./non-gemm-out"):
    entries = os.listdir(path)
    print(entries)
    # Filter only directories
    directories = [entry for entry in entries if os.path.exists(os.path.join(path, entry))]
    print (directories)
    return directories

def avg_df (dfs): 
    print("averaging")
    df_ = pd.concat([df for df in dfs])
    df_ = df_.sum().to_frame()
    print (type(df_))
    print(df_.head())
    return df_

def summarize_data(profile_dir_:str="test_rachid_ispass_inference_data", csv_dir:str="test_rachid_ispass_summary_csv"):
    models = [
        # 'swin-t',
        # 'swin-s',
        # 'swin-b',
        'fasterrcnn',
        # 'maskrcnn',
        # 'segformer-b0',
        # 'gpt2',
        # 'gpt2-large',
        # 'gpt2-xl',
        # 'llama'
        ]
    batch_sizes =[1,8]
    devices = ['cpu', 'cuda']
    for dev in devices:
        os.system(f'mkdir -p {csv_dir}/{dev}')
        profile_dir = f"{profile_dir_}_{dev}"
        for model in models: 
            print("RACHID")

            for n in batch_sizes:
                df_s = []
                profile_data_dir = f'{profile_dir}/{model}/batch_size{n}'
                files = get_files(profile_data_dir)
                print("RACHID")
                print(files)
                model_name = f"{model}_{n}"
                drop_first = False
                for file in files: 
                    file_ = f"{profile_data_dir}/{file}"
                    print("RACHID")
                    print(file)
                    #df_ = remove_warmup(profile_data_dir,file_, csv_dir, drop_first)
                    df_ = pd.read_json(file_)
                    df_summarized = summarize_onnx_profile(df_, model_name)
                    if (df_summarized.empty): 
                        continue
                    df_summarized.to_csv('tmp.csv')
                    print(df_summarized.head())
                    df_s.append(df_summarized)
                    drop_first = False
                df_avg = avg_df(df_s) 
                df_avg.to_csv(f'{csv_dir}/{dev}/{model_name}.csv') 
                
                 
             
     

        # data_file = remove_warmup(profile_data_file, profile_dir)
        # df_file_stats = summarize_onnx_profile(data_file)

    # df_prefill = summarize_onnx_profile(prefill_file)
    # df_decode = summarize_onnx_profile(decode_file)

    # df_prefill.to_csv(f"{csv_dir}/{model_prefix}_prefill.csv")
    # df_decode.to_csv(f"{csv_dir}/{model_prefix}_decode.csv")

def debug(): 
    summarize_data()    

def main(): 
    parser = argparse.ArgumentParser(description ='Gen Configuration')
    
    parser.add_argument('--model_name', dest ='model_name', 
                        metavar ='model_name', required=True,
                        help = "Model Name")
    parser.add_argument('--profile_dir', metavar ='profile_dir', 
                    required = True, dest ='profile_dir', 
                    type = str, 
                    help ='Path to directory containing the onnx profiling JSON data file.')
    parser.add_argument('--seq_len', metavar ='seq_len', 
                    required = True, dest ='seq_len', type= int, 
                    help ='Input Sequence Length')
    parser.add_argument('--tokens', metavar ='tokens', 
                    required = True, dest ='tokens', type= int, 
                    help ='Number of tokens to be generated')

    parser.add_argument('--out_dir', metavar ='out_dir', 
                    required = True, dest ='out_dir', 
                    type = str, 
                    help ='Output Directory to store summarized data in csv format')
    
    parser.add_argument('--profiling_data', metavar ='profiling_data', 
                    required = False, dest ='profiling_data', 
                    type = str, 
                    help ='Name of the onnx profiling JSON data file.')
    args = parser.parse_args()
    
    seq_len = args.seq_len
    max_new_tokens = args.tokens 
    model_name = args.model_name
    out_dir = f"{args.out_dir}/"
    profile_dir = f"{args.profile_dir}"
    prefix = f'{model_name}_{seq_len}_{max_new_tokens}'
    profile_file = glob.glob(f"{profile_dir}/{prefix}_*")[0] if args.profiling_data is None else args.profiling_data
    print (profile_file)
    print (len(glob.glob(f"{profile_dir}/{prefix}_*")))
    if (len(glob.glob(f"{profile_dir}/{prefix}_*")) > 1): 
        print ("We can process only 1 file at a time.")
        print ("Please enter the filename as an argument")
        exit()
    
    os.system(f'mkdir -p {out_dir}/{prefix}')
    summarize_data(prefix, profile_file, profile_dir, out_dir)

if __name__ == "__main__": 
    debug()
