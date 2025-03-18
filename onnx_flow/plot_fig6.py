import pandas as pd 
import json
import glob
from matplotlib import pyplot as plot
import os
import numpy as np 

dir = ['cpu', 'cuda']
classification = {"classification":["swin-b", "swin-s", "swin-t", 'resnet50', 'mobilenet',]}#"vit-b16", "vit-l16", ]}
#classification = {"classification":["swin-base",]}
detection = {"detection":[ "fasterrcnn","maskrcnn", ]}#"detr","detr-1", "detr-2",
segmentation = {"segmentation":[ "segformer-b0",]} #"maskformer",
language_models = {"language_models":["bert", "gpt2","gpt2-large", ]} # "gpt2-large",
large_language_models = {"large_language_models":[ "gpt2-xl",  "llama", ]}
batch_sizes = [1]#[1,8]

_cpu_files = glob.glob('data_csv/cpu/*')
cpu_files = [x.split('/')[-1] for x in _cpu_files]
print (f'len cpu_files:{len(cpu_files)}')



_cuda_files = glob.glob('data_csv/cuda/*')
cuda_files = [x.split('/')[-1] for x in _cuda_files]
print (f'len cuda_files:{(cuda_files)}')


out_dir = 'server_figures'
os.system(f'mkdir -p {out_dir}')


colors = {"gemm":'#4C443C' , "non_gemm":'#DEB841', "normalization":"#DEB841", "activation":"#769FB6", "arithmetic":"#D16666",  "memory":"#55917F", "Other":"#646E78", "comm":"#FAB3A9", "pooling":"#BDBBB6",  "logit_computation":"#254E70", "roi":"#FAE8EB", }#"Embedding":"#83D628", "get":"#646E78", "Interpolate":"#994FC3",
groups = ['activation', 'normalization', 'logit_computation', 'memory', 'roi', 'pooling', 'non_deterministic', 'comm','arithmetic' ]
plot_grps = ['activation', 'normalization', 'logit_computation', 'memory', 'roi', 'pooling', 'non_deterministic', 'comm','arithmetic', 'gemm' ]
figsize = (2,4)
width = 0.5

def _plot_gemm_vs_non_gemm (cpu, gpu, figsize, width, out_dir, model, normalize = True): 
    df_cpu = pd.read_csv (cpu)
    df_gpu = pd.read_csv (gpu)
    
    df_cpu = df_cpu[['Ops', 'dur']]
    df_gpu = df_gpu[['Ops', 'dur']]

    #df_cpu_pct = df_cpu[df_cpu['Ops'].isin(['gemm','non_gemm'])].set_index('Ops')
    #df_gpu_pct = df_gpu[df_gpu['Ops'].isin(['gemm','non_gemm'])].set_index('Ops')
    df_cpu_pct = df_cpu[df_cpu['Ops'].isin(plot_grps)].set_index([ 'Ops'])
    df_gpu_pct = df_gpu[df_gpu['Ops'].isin(plot_grps)].set_index('Ops')
    

    if (normalize):
        df_cpu_pct = get_col_pct(df_cpu_pct,'dur')
        df_gpu_pct = get_col_pct(df_gpu_pct,'dur')
    
    df_cpu_pct.rename(columns={'dur': 'cpu'}, inplace=True)
    
    try:
        #df_cpu_pct.transpose().to_csv(f'{csv_dir}/cpu_{file}')
        print (file)
    except Exception as e: 
        print(e)
    
    df_gpu_pct.rename(columns={'dur': 'cpu+gpu'}, inplace=True)
    
    try:
        #df_gpu_pct.transpose().to_csv(f'{csv_dir}/gpu_{file}')
        print (file)
    except Exception as e: 
        print(e)
           

    df_plot = pd.concat((df_cpu_pct.transpose(), df_gpu_pct.transpose()))
    

    plt = df_plot.plot.bar(stacked=True,legend = True, figsize = figsize, width = width, color = colors)

    plt.set_ylabel(cpu.split('/')[-1])
    plot.savefig(f"{out_dir}/{cpu.split('/')[-1]}.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    return df_plot

def plot_gemm_vs_non_gemm (out_dir, figsize, width,model): 
    for file in cpu_files: 
        if (file in cuda_files): 
            _plot_gemm_vs_non_gemm(f'data_csv/cpu/{file}', f'data_csv/cuda/{file}', figsize, width, out_dir,file)
        else: 
            print (f'{file}: No GPU data!, {file}')


def get_col_pct(df,col): 
    df[col] = df[col] / (df[col].sum())
    return df



#task= large_language_models
#task_ = 'large_language_models'
#device = 'cuda'
#out_dir = f'server_figures/breakdown'
#os.system(f'mkdir -p {out_dir}')
def _concat_task (_file,task,task_,device,out_dir):
    print (task_)
    models = task [task_]
    df_task = pd.DataFrame()
    index = None
    order = None
    for _model in models:
        for batch in batch_sizes:

            file = f'{_file}/{device}/{_model}_{batch}.csv'
            print(f"Fiel name {file}")
            if (os.path.exists(file)):
                df = pd.read_csv(file)
                df.rename(columns={'0': 'dur'}, inplace=True)
                # print (df.head())
                # return 
            else: 
                print(f"FILE {_file}/{device}/{_model}_{batch}.csv NOT FOUND")
                break
            model = f'{_model}-{batch}' 
            df = df[~df['Ops'].isin(['non_gemm','total']) ]
            
            df[model] = (df ['dur'] / df['dur'].sum()) * 100
            df = df[['Ops', model]]
            df.set_index('Ops', inplace = True)
            df.sort_values (model,ascending = False ,inplace=True)
            gemm_row = df.loc['gemm'].to_frame().transpose()
            df.drop('gemm',inplace = True)
            order = df.index.tolist()
            if (not index is None): 
                df.reindex (index)
                order = index
            df_ = pd.concat((gemm_row, df)).transpose()
            #return df_, order
            df_task = pd.concat((df_task,df_))
            
        index = order
    
    if (not (df_task.empty)):
        figsize = (6,4) if task_ == 'large_language_models' else (4,6)
        plt = df_task.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = colors)#figsize = (6,4)
        
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        plt.set_xlabel('')
        plot.savefig(f"{out_dir}/models-breakdown.png", format="png", bbox_inches="tight", dpi=300)
        df_task.head(10)
        new_row ={}
        for col in df_task.columns.tolist(): 
            new_row [col] = df_task[col].mean() 
        df_task.loc[task_] = new_row
        df_task.to_csv(f'{out_dir}/{task_}_breakdown.csv')
        return df_task

    

def concat_task (file,tasks, _out_dir, devices): 

    for device in devices:
        df_ = pd.DataFrame()
        for task in tasks: 
            task_= list(task.keys())[0]
            out_dir = f'{_out_dir}/{device}/breakdown/{task_}'
            os.system(f'mkdir -p {out_dir}')
            df = _concat_task(file,task, task_, device,out_dir)
            if (not(df is None)):
                print ('hellloa')
                print (df_.head())
                print (df.head())
                df_= pd.concat((df_,df.loc[task_].to_frame().transpose()))
        df_.to_csv(f'{_out_dir}/{device}_summary.csv')

def get_machine_comparison (machines,device): 
    devices = [device]
    df_ = pd.DataFrame()
    for machine in machines: 
        for device in devices: 
            dir = f'{machine}_figures'
            file = f'{dir}/{device}_summary.csv'
            df = pd.read_csv(file)
            df['machine'] = machine 
        df_ = pd.concat((df_,df))
             
         
    print(df_['machine'].unique().tolist())
    return df_

def _plot_machine_comparison (df,out_dir,device):
    
    os.system(f'mkdir -p {out_dir}')
    df.rename(columns={'Unnamed: 0': 'task'}, inplace=True)
    #df.head()
    df.set_index('machine',inplace=True)
    df = df.groupby('task').apply(lambda x: x)
    df.plot.bar(stacked =True,linewidth=3)
    tasks =df['task'].unique().tolist()
    #df.head(20)
    for task in tasks:
        plt = df.loc[task:task].plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = colors)#figsize = (6,4)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        plot.xlabel("")
        plot.savefig(f"{out_dir}/{device}_no_labels_{task}_hw_comparison.png", format="png", bbox_inches="tight", dpi=300)

def plot_machine_comparison (device):
    df = get_machine_comparison(['mobile','server'],device)
    _plot_machine_comparison (df, 'hw_comparison',device) 
    

def plot_fig6(task = "llm", models = None, batch_sizes=[1], prof_dir = "summary-onnx", out_dir = "./fig6_onnx", device = "cuda"): 
    
    df_task = pd.DataFrame()
    index = None
    order = None
    task_ = task
    for _model in models:
        for batch in batch_sizes:

            file = f'{prof_dir}/{device}/{_model}_batch_size{batch}.csv'
            print(f"Fiel name {file}")
            if (os.path.exists(file)):
                df = pd.read_csv(file)
                df.rename(columns={'0': 'dur'}, inplace=True)
                # print (df.head())
                # return 
            else: 
                print(f"FILE {file} NOT FOUND")
                break
            model = f'{_model}-{batch}' 
            df = df[~df['Ops'].isin(['non_gemm','total']) ]
            
            df[model] = (df ['dur'] / df['dur'].sum()) * 100
            df = df[['Ops', model]]
            df.set_index('Ops', inplace = True)
            df.sort_values (model,ascending = False ,inplace=True)
            gemm_row = df.loc['gemm'].to_frame().transpose()
            df.drop('gemm',inplace = True)
            order = df.index.tolist()
            if (not index is None): 
                df.reindex (index)
                order = index
            df_ = pd.concat((gemm_row, df)).transpose()
            #return df_, order
            df_task = pd.concat((df_task,df_))
            
        index = order
    
    if (not (df_task.empty)):
        #figsize = (6,4) if task_ == 'large_language_models' else (4,6)
        plt = df_task.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = colors)#figsize = (6,4)
        
        plt.tick_params(labelbottom=True)
        plt.tick_params(labelleft=True)
        plt.set_xlabel('')
        plot.savefig(f"{out_dir}/fig6_{task_}_breakdown.png", format="png", bbox_inches="tight", dpi=300)
        df_task.head(10)
        new_row ={}
        for col in df_task.columns.tolist(): 
            new_row [col] = df_task[col].mean() 
        df_task.loc[task_] = new_row
        df_task.to_csv(f'{out_dir}/{task_}_breakdown.csv')
       #return df_task


def main(): 
    prof_dir = "./summary-onnx"

    models = [ "gpt2-xl",  "llama", ]

    batch_sizes = [1]

    task = "llm"

    device = "cuda"

    out_dir = "./fig6_onnx"
    if not (os.path.exists(out_dir)): 
        os.system(f"mkdir -p {out_dir}")
    plot_fig6(task= task, models = models, batch_sizes=batch_sizes, prof_dir=prof_dir, out_dir=out_dir, device = device)

    
def plot_machine(): 
    #plot_machine_comparison('cuda')
    machine = 'server' # 'server'#  
    machines = ['server','mobile']
    out_dir_ = 'rachid-onnx-a100'
    for machine in machines:
        #file = f'ispass_summary_csv_{machine}'
        file = f"rachid_ispass_summary_csv"
        devices = ['cuda', 'cpu']
        tasks = [classification, detection,  language_models, segmentation, large_language_models, segmentation ]
        #tasks = [detection]
        out_dir = f'{machine}_figures_2'
        concat_task(file,tasks, out_dir_, devices)




if __name__ =='__main__': 
    main()