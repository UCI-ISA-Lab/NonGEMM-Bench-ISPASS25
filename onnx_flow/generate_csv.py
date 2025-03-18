import pandas as pd 
import json 
import glob 
import argparse
import os

non_gemm_cum = ['ReorderInput','ReorderOutput','Mod', 'ScatterND', 'Not', 'GlobalAveragePool','ReduceMax', 'Reciprocal', 'ReduceMin', 'Min', 'Floor', 'Ceil', 'Pad', 'Range', 'Clip', 'Exp', 'TopK', 'Max', 'GreaterOrEqual', 'And', 'NonZero', 'ReduceProd', 'NonMaxSuppression', 'If', 'Sqrt', 'Log', 'RoiAlign', 'ScatterElements', 'Greater',  'Loop','ReduceMean', 'SimplifiedLayerNormalization', 'Gelu' ,'Tile', 'ConstantOfShape', 'MaxPool', 'Relu', 'Resize', 'CumSum', 'Cos', 'Sin', 'Flatten', 'Equal', 'Expand', 'Sigmoid','Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Add', 'LayerNormalization', 'Slice', 'Squeeze', 'Split', 'Transpose', 'Sub', 'Cast', 'Pow', 'Div', 'Where', 'Mul', 'Softmax', 'FastGelu','MemcpyToHost', 'MemcpyFromHost', 'BiasGelu', 'Neg', 'Less', 'QuickGelu']
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

def summarize_onnx_profile (file): 
    df = pd.read_json(file)
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
    providers = df['provider'].unique().tolist()
    df_gemm = df[df['op_name'].isin(gemm)]
    df_non_gemm = df[df['op_name'].isin(non_gemm_cum)]
    total = df['dur'].sum()
    total_gemm = df_gemm ['dur'].sum()
    total_non_gemm = df_non_gemm['dur'].sum()
    print (f'Total Inference time: {total} us')
    print (f'Total GEMM time: {total_gemm} us, {total_gemm / total * 100} %')
    print (f'Total non-GEMM time: {total_non_gemm} us, {total_non_gemm / total * 100} %')
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
    #df_total_stats.loc[len(df_total_stats)] = {'dur':df_non_deterministic['dur'].sum(), 'Ops':'non_deterministic'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_comm['dur'].sum(), 'Ops':'comm'}
    df_total_stats.loc[len(df_total_stats)] = {'dur':df_arithmetic['dur'].sum(), 'Ops':'arithmetic'}

    return df_total_stats 

def avg_dfs (dfs, key, value):
    key_all = all(df[key].equals(dfs[0][key]) for df in dfs)
    if not key_all:
        raise ValueError("Key columns are not equal or not in the same order in all DataFrames.")
    
    # Combine the value columns
    print ((dfs[0]).head())
    values_combined = pd.concat([df[value] for df in dfs], axis=1)
    
    print ((values_combined).head())

    # Calculate the average of the value columns
    values_avg = values_combined.mean(axis=1) 
    
    # Create the new DataFrame with the key column and averaged values
    df_avg = pd.DataFrame({
        key: dfs[0][key],
        value: values_avg
    })
    return df_avg


def avg_profiles (pwd, model_name, batch_size, out_dir, device, key='Ops', value="dur"): 
    files = glob.glob(f'{pwd}/*.json')
    print (len(files))
    dfs = []
    for file in files: 
        df = summarize_onnx_profile(file)
        dfs.append(df)
    os.system(f'mkdir -p {out_dir}/{device}')
    name = f'{out_dir}/{device}/{model_name}_{batch_size}.csv'
    print(len(dfs))
    df_avg = avg_dfs (dfs, key, value)
    df_avg.to_csv(name)

    
def postprocess(profile_dir, out_dir, device): 
    models = glob.glob(f'{profile_dir}/*')
    print(f'Models: {models}')
    for model_name in models: 
        batch_sizes = glob.glob(f'{model_name}/*')
        print (f'batches: {batch_sizes}')
        
        for i, batch_size in enumerate(batch_sizes): 
            pwd = f'{batch_size}'
            avg_profiles(pwd, model_name.split('/')[-1], batch_size.split('/')[-1], out_dir, device)

if __name__=="__main__": 
    parser = argparse.ArgumentParser(description ='Post Processing Collected Profiles')
    
    parser.add_argument('--profile_dir', dest ='profile_dir', 
                        metavar ='profile_dir', nargs ='*', required=True,
                        help = "directory where all the profiles were stored")  

    
    parser.add_argument('--out_dir', metavar ='out_dir', 
                    required = True, dest ='out_dir', 
                    action ='append', 
                    help ='output directory to store the generated csv files for easier access')
    parser.add_argument('--device', metavar ='device', 
                    required = True, dest ='device', 
                    action ='append', 
                    help ='Backend Device to Run the models')
    args = parser.parse_args()
    profile_dir = args.profile_dir [0]
    out_dir = args.out_dir[0]
    device = args.device[0]

    postprocess(profile_dir, out_dir, device)
    exit()
    try: 
        postprocess(profile_dir, out_dir, device)
    except: 
        print ("PostProcessing Failed!")
        pass
                
## python generate_csv.py --profile_dir new_inference_data_cuda --out_dir ispass_new_data --device cuda > logs/post.log
## python generate_csv.py --profile_dir new_inference_data_cpu --out_dir ispass_new_data --device cpu > logs/post.log
