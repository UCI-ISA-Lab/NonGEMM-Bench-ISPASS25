import torch 
import transformers 
import os 
import subprocess
import gc
import signal
import argparse 
import datasets
#import torchvision 
from torch.utils.data import Subset
from torch.utils.data import DataLoader 
import random
import time
from typing import List
#from torchvision import datasets
import pandas as pd 
torch.manual_seed(1969)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

gemm_ops = ["aten::mm", "aten::matmul", "aten::bmm", "aten::linear", 
       "aten::addmm", "aten::addbmm", "aten::baddbmm", "aten::mv","aten::dot",       
       "aten::ger", "aten::matmul.out", "aten::scaled_dot_product_attention", 
       "aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::conv_tbc", 
       "aten::conv_transpose1d", "aten::conv_transpose2d", "aten::conv_transpose3d", 
       "aten::slow_conv3d", "aten::slow_conv_dilated2d", "aten::slow_conv_dilated3d", 
       "aten::slow_conv_transpose2d", "aten::slow_conv_transpose3d", "aten::thnn_conv2d", 
       "aten::thnn_conv_depthwise2d","aten::scaled_dot_product_attention", "aten::linear",
       'wqlinearmmfunction',
       "conv1d", 'aten::_native_multi_head_attention', 
       'segformerdwconv', 'aten::einsum', 'aten::_scaled_dot_product_efficient_attention', 
       "aten::convolution", "aten::_scaled_dot_product_flash_attention",  "cutlass::Kernel2",
       ]

# Ops made of multiple aten operators to be captured 
ops = ["conv1d", "wqlinear_gemm", "llamarmsnorm"]


def debug_test_aggregate (prof, op, filename): 

    result_rows = [] 
    for op in ops: 
        pass


def test_aggregate(prof, ops, filename): 
    st = time.perf_counter()
    reshape = 0
    linear = 0 
    matmul = 0
    # Prepare the result list
    result_rows = []
    op_dict = {} 


    # Process each operation
    for op in ops:
        op_rows = []
        for e in prof.profiler.function_events:
            if e.name == "Inference_prof" : #'torch._C._autograd.DeviceType'
                #print (e.device_type)
                #print (type(e.device_type))
                if (e.device_type == torch._C._autograd.DeviceType.CPU):
                    op_rows.append({
                        'name': e.name.lower(),
                        'cpu_time (us)': e.cpu_time,
                        'cuda_time (us)': e.device_time,
                        'total_time (us)': e.cpu_time + e.cuda_time,
                        'count':1
                    })
            elif e.cpu_parent and "_prof" in e.cpu_parent.name and e.name == op:
                op_rows.append({
                    'name': e.name.lower(),
                    'cpu_time (us)': e.cpu_time,
                    'cuda_time (us)': e.cuda_time,
                    'total_time (us)': e.cpu_time + e.cuda_time, 
                    'count':1
                })
                if (e.name=="aten::reshape"):
                    reshape += 1 
                elif (e.name=="aten::linear" ): 
                    linear+=1
                elif (e.name=="aten::matmul"):
                    matmul+=1

        
        # Aggregate the operation results
        df = pd.DataFrame(op_rows)
        cpu = df['cpu_time (us)'].sum() if not df.empty else 0
        cuda = df['cuda_time (us)'].sum() if not df.empty else 0
        count = df['count'].sum() if not df.empty else 0
        result_rows.append({
            'name': op.lower(),
            'cpu_time (us)': cpu,
            'cuda_time (us)': cuda,
            'total_time (us)': cpu + cuda, 
            'count': count
        })
    
    # Create the final DataFrame
    df_ = pd.DataFrame(result_rows)
    
    # Save to CSV
    df_.to_csv(filename)
    
    et = time.perf_counter()
    print(f"Time to Summarize Files {et - st} s")
    return

def aggreagate (prof, ops, filename): 
    print (f"Aggregating profiles:")
    st = time.perf_counter()
    columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'count']
    df_= pd.DataFrame(columns=columns)

    for op in ops:
        df = pd.DataFrame(columns=columns)
        skip_first = False
        cpu = 0
        cuda = 0 
        count = 0
        kernel = False
        if ("bit_8_" in op): 
            kernel = True
            if ("bit_8_kernel_" in op): 
                op_ = op.replace("bit_8_kernel_","")
            elif("bit_8_quantize_" in op): 
                op_ = op.replace("bit_8_quantize_","")
            elif("bit_8_dequantize_" in op): 
                op_ = op.replace("bit_8_dequantize_","")
            else: 
                pass
        
        for e in prof.profiler.function_events: 
            if (kernel): 
                if (e.name == op_): 
                    cpu += e.cpu_time
                    cuda += e.device_time
                    count += 1
            else:
                if not(e.cpu_parent is None):
                    if (e.cpu_parent.name == "Inference_prof" or e.name=="Inference_prof" or e.cpu_parent.name =="aten::multinomial") and (e.name == op) and (e.name!="aten::multinomial"):
                        cpu += e.cpu_time
                        cuda += e.device_time
                        count += 1
                    if ("8bit" in e.cpu_parent.name):
                        op_ = op.replace("_q8bit","") 
                        if (e.name ==op_): 
                            cpu += e.cpu_time
                            cuda += e.cuda_time
                            count += 1


        new_entry_ = {'name':op.lower(), 'cpu_time (us)':[cpu], 'cuda_time (us)':[cuda], 'total_time (us)':[cpu+cuda], 'count':[count] }
        df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    df_.to_csv(filename)
    et = time.perf_counter() 
    print (f"Finished Aggregating Profiles: {et - st} s")
    return     

def generate_report (filename): 
    print ("Generating CSV Files")
    st = time.perf_counter()
    df = pd.read_csv(filename)
    df = df.drop(df.columns[0], axis=1)
   

    inference = df[df['name'].str.contains('Inference_')]

    gemm_ops_ = []
    non_gemm_ops_ = []
    uniq = df['name'].unique().tolist()
    for i in uniq: 
        if (i != "inference_prof"):
            if ("_prof" in i):
                for j in gemm_ops: 
                    if (i.replace('_prof',"") in j):
                        gemm_ops_.append(i)
                        break 
            else: 
                if ("convolution" in i) or ("cutlass::kernel2") in i: 
                    gemm_ops_.append(i)
                else:
                    non_gemm_ops_.append(i)
      
    gemm_ops_ = gemm_ops_ + gemm_ops
    # print (gemm_ops_)
    # print ("GEMM OPS DONE")

    gemm = df[df ["name"].isin(gemm_ops_)]
    non_gemm = df[~df["name"].isin(gemm_ops_)]
    non_gemm = non_gemm[~ non_gemm['name'].str.contains('profiler|inference|cuda')]
   
    new_gemm_entry = {'name':"GEMM", 'cpu_time (us)':[gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[gemm['cuda_time (us)'].sum()], 'total_time (us)':[gemm['total_time (us)'].sum()]} 
    
    new_non_gemm_entry = {'name':"NonGEMM", 'cpu_time (us)':[non_gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[non_gemm['cuda_time (us)'].sum()], 'total_time (us)':[non_gemm['total_time (us)'].sum()]} 
    
    new_inference_entry = {'name':"Inference", 'cpu_time (us)':[inference['cpu_time (us)'].sum()], 'cuda_time (us)':[inference['cuda_time (us)'].sum()], 'total_time (us)':[inference['total_time (us)'].sum()]} 

    df = pd.concat((df, pd.DataFrame(new_gemm_entry)))
    df = pd.concat((df, pd.DataFrame(new_non_gemm_entry)))
    
 
    gemm = pd.concat((gemm, pd.DataFrame(new_gemm_entry)))
    gemm = pd.concat((gemm, pd.DataFrame(new_inference_entry)))
    
    non_gemm = pd.concat((non_gemm, pd.DataFrame(new_non_gemm_entry)))
    non_gemm = pd.concat((non_gemm, pd.DataFrame(new_inference_entry)))

    gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/gemm.csv")
    non_gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/non_gemm.csv")
    df.to_csv(f'{filename}')
    et = time.perf_counter() 
    print (f"Finished generating Report {et - st} s")

def _analyze_prof (prof, filename, custom): 
    print ("Analyzing Profile logs")
    #print (prof.key_averages())
    avg = prof.key_averages() 
    ops = []
    bit_8_mm_kernel = "cutlass::Kernel2"
    bit_8_quantize = "kInt8VectorQuant"
    bit_8_dequantize = "kdequant_mm_int32_fp16"
    if "8bit" in filename: 
        for a in avg: 
            if (bit_8_mm_kernel in a.key): 
                k = f"bit_8_kernel_{a.key}"
                ops.append(k)
            if (bit_8_quantize in a.key): 
                k = f"bit_8_quantize_{a.key}"
                ops.append(k)
            if (bit_8_dequantize in a.key): 
                k = f"bit_8_dequantize_{a.key}"
                ops.append(k)
    
    for a in avg: 
        if ("Step" in a.key): 
            ops.append(a.key)
        if ("Inference_" in a.key): 
            ops.append(a.key)

        multinomial = True
        if not (a.cpu_parent is None):
            if ("Inference_" in a.key ): 
                for child in a.cpu_children: 
                    
                    if not (child.name in ops):
                        #ops.append(child.name)
                        if ("8bit" in child.name):
                            ops.append(child.name) 
                            for child8 in child.cpu_children:
                                child8_ = f"{child8.name}_q8bit"
                                if not (child8_ in ops):
                                    ops.append(child8_)
                        else:
                            if (child.key == "aten::multinomial") and multinomial: 
                                for child_ in child.cpu_children:
                                    if not(child_ in ops):
                                        ops.append(child_.name)
                                multinomial = False 
                            else:
                                if (child.key != "aten::multinomial"):
                                    ops.append(child.name)
                inf =  a
                break
    
    if (custom): 
        generate_report(filename)

    else: 
        print (ops)
        
        aggreagate (prof, ops, filename)
        
        generate_report(filename)
    return


########### Shape Recording ####################
def aggreagate_shape (prof, ops_to_be_recorded, filename): 
    
    print (f"Aggregating profiles:")
    st = time.perf_counter()
    columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'shape', 'count']
    df_= pd.DataFrame(columns=columns)
    avg = prof.key_averages(group_by_input_shape=True)
    print (ops_to_be_recorded)
    for op in avg: 
        if op.key in ops_to_be_recorded:
            shape = str(op.input_shapes)
            count = op.count
            mem_cuda = op.device_memory_usage 
            mem_cpu = op.cpu_memory_usage 
            new_entry_ = {'name':op.key.lower(), 'count':[count], 'shape':shape, 'mem_cpu':mem_cpu, 'mem_cuda':mem_cuda }
            df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    df_.to_csv(filename)
    et = time.perf_counter() 
    print (f"Finished Aggregating Profiles: {et - st} s")
    return
    
    # print (f"Aggregating profiles:")
    # st = time.perf_counter()
    # columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'shape', 'count']
    # df_= pd.DataFrame(columns=columns)

    # for op in ops:
    #     df = pd.DataFrame(columns=columns)
    #     skip_first = False
    #     cpu = 0
    #     cuda = 0 
    #     shape = []
    #     count = 0
    #     for e in prof.profiler.function_events: 
            
    #         if not(e.cpu_parent is None):
    #             if (e.cpu_parent.name == "Inference_prof" or e.name=="Inference_prof") and (e.name == op): 
    #                 shape.append(e.ipnut_shapes)
    #                 count += 1
    #             if ("8bit" in e.cpu_parent.name):
    #                 op_ = op.replace("_q8bit","") 
    #                 if (e.name ==op_): 
    #                     # new_entry = {'name':f"{e.name.lower()}_q8bit", 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
    #                     # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True)
    #                     cpu += e.cpu_time
    #                     cuda += e.cuda_time
    #                     count += 1

    #     # cpu = df['cpu_time (us)'].sum()
    #     # cuda = df['cuda_time (us)'].sum()
    #     # count = df['count'].sum()
    #     # del df 
    #     new_entry_ = {'name':op.lower(), 'count':[count], 'shape':shape }
    #     df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    # df_.to_csv(filename)
    # et = time.perf_counter() 
    # print (f"Finished Aggregating Profiles: {et - st} s")
    # return     

def generate_report_shape (filename): 
    print ("Generating CSV Files")
    st = time.perf_counter()
    df = pd.read_csv(filename)
    df = df.drop(df.columns[0], axis=1)
   

    inference = df[df['name'].str.contains('Inference_')]

    gemm_ops_ = []
    non_gemm_ops_ = []
    uniq = df['name'].unique().tolist()
    for i in uniq: 
        if (i != "inference_prof"):
            if ("_prof" in i):
                for j in gemm_ops: 
                    if (i.replace('_prof',"") in j):
                        gemm_ops_.append(i)
                        break 
            else: 
                non_gemm_ops_.append(i)
        
    gemm_ops_ = gemm_ops_ + gemm_ops
    # print (gemm_ops_)
    # print ("GEMM OPS DONE")

    gemm = df[df ["name"].isin(gemm_ops_)]
    non_gemm = df[~df["name"].isin(gemm_ops_)]
    non_gemm = non_gemm[~ non_gemm['name'].str.contains('profiler|inference')]
   
    #new_gemm_entry = {'name':"GEMM", 'cpu_time (us)':[gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[gemm['cuda_time (us)'].sum()], 'total_time (us)':[gemm['total_time (us)'].sum()]} 
    
    #new_non_gemm_entry = {'name':"NonGEMM", 'cpu_time (us)':[non_gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[non_gemm['cuda_time (us)'].sum()], 'total_time (us)':[non_gemm['total_time (us)'].sum()]} 
    
    #new_inference_entry = {'name':"Inference", 'cpu_time (us)':[inference['cpu_time (us)'].sum()], 'cuda_time (us)':[inference['cuda_time (us)'].sum()], 'total_time (us)':[inference['total_time (us)'].sum()]} 

    #df = pd.concat((df, pd.DataFrame(new_gemm_entry)))
    #df = pd.concat((df, pd.DataFrame(new_non_gemm_entry)))
    
 
    #gemm = pd.concat((gemm, pd.DataFrame(new_gemm_entry)))
    #gemm = pd.concat((gemm, pd.DataFrame(new_inference_entry)))
    
    #non_gemm = pd.concat((non_gemm, pd.DataFrame(new_non_gemm_entry)))
    #non_gemm = pd.concat((non_gemm, pd.DataFrame(new_inference_entry)))

    gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/gemm.csv")
    non_gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/non_gemm.csv")
    df.to_csv(f'{filename}')
    et = time.perf_counter() 
    print (f"Finished generating Report {et - st} s")

def _analyze_prof_shape (prof, filename, ops_to_be_recorded): 
    print ("Analyzing Profile logs")
    #print (prof.key_averages())
    avg = prof.key_averages(group_by_input_shape=True) 
    with open("shape_stat.txt", "w") as f:
        print(avg, file=f)
    
    ops = []
    for a in avg: 
        if ("Step" in a.key): 
            ops.append(a.key)
        if ("Inference_" in a.key): 
            ops.append(a.key)
            
        
        # if custom:
        #     if not (a.cpu_parent is None): 
        #         if ("_prof" in a.cpu_parent.name):
        #             if not (a.key in ops): 
        #                 ops.append(a.key)
        #         else: 
        #             if ("_prof" in a.key) and not (a.key in ops): 
        #                 ops.append(a.key)
                    
                    
        multinomial = True
        if not (a.cpu_parent is None):
            if ("Inference_" in a.key ): 
                for child in a.cpu_children: 
                    
                    if not (child.name in ops):
                        #ops.append(child.name)
                        if ("8bit" in child.name):
                            ops.append(child.name) 
                            for child8 in child.cpu_children:
                                child8_ = f"{child8.name}_q8bit"
                                if not (child8_ in ops):
                                    ops.append(child8_)
                        
                        else:
                            if (child.key == "aten::multinomial") and multinomial: 
                                for child_ in child.cpu_children:
                                    if not(child_ in ops):
                                        ops.append(child_.name)
                                multinomial = False 
                            else:
                                if (child.key != "aten::multinomial"):
                                    ops.append(child.name)
                inf =  a
                break

    print (ops)
    
    aggreagate_shape (prof, ops, filename)
    #test_aggregate(prof, ops, filename)
    
    generate_report_shape(filename)
    return

########## End Shape Recording ################

######### DYNAMO ###################

def _analyze_prof_dynamo (prof, filename, custom): 
    print ("Analyzing Profile logs")
    #print (prof.key_averages())
    avg = prof.key_averages() 
    ops = []
    for a in avg: 
        # if ("Step" in a.key): 
        #     ops.append(a.key)
        # if ("Inference_" in a.key): 
        #     ops.append(a.key)
            
        
        # if custom:
        #     if not (a.cpu_parent is None): 
        #         if ("_prof" in a.cpu_parent.name):
        #             if not (a.key in ops): 
        #                 ops.append(a.key)
        #         else: 
        #             if ("_prof" in a.key) and not (a.key in ops): 
        #                 ops.append(a.key)
                    
                    
        multinomial = True
        if not (a.cpu_parent is None):
            if ("Inference_prof_7" in a.key): 
                for child in a.cpu_children: 
                    if ("Torch-Compiled Region" in child.key):
                        for child_ in child.cpu_children: 
                                if ("CompiledFunction" in child_.key):
                                    for child_op in child_.cpu_children:
                                        if not (child_op.name in ops):
                                            ops.append(child_op.name)
                                    break                      
                inf =  a
                break
    if (custom): 
        #aggregate_custom(prof, ops, filename)
        #test_aggregate(prof, ops, filename)
        generate_report(filename)

    else: 
        print (ops)
        
        aggreagate_dynamo (prof, ops, filename)
        #test_aggregate(prof, ops, filename)
        
        generate_report(filename)
    return


def aggreagate_dynamo (prof, ops, filename): 
    print (f"Aggregating profiles:")
    st = time.perf_counter()
    columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'count']
    df_= pd.DataFrame(columns=columns)

    for op in ops:
        df = pd.DataFrame(columns=columns)
        skip_first = False
        cpu = 0
        cuda = 0 
        count = 0
        for e in prof.profiler.function_events: 
            
            if not(e.cpu_parent is None):
                if (e.cpu_parent.name == "CompiledFunction") and (e.name == op):# and (not (e.cpu_parent.cpu_parent.name in [f'Inference_prof_{i}' for i in range (5)])):## surgery multinomial 
                    # new_entry = {'name':e.name.lower(), 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
                    # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True) 
                    cpu += e.cpu_time
                    cuda += e.device_time
                    count += 1
                if ("8bit" in e.cpu_parent.name):
                    op_ = op.replace("_q8bit","") 
                    if (e.name ==op_): 
                        # new_entry = {'name':f"{e.name.lower()}_q8bit", 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
                        # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True)
                        cpu += e.cpu_time
                        cuda += e.cuda_time
                        count += 1

        # cpu = df['cpu_time (us)'].sum()
        # cuda = df['cuda_time (us)'].sum()
        # count = df['count'].sum()
        # del df 
        new_entry_ = {'name':op.lower(), 'cpu_time (us)':[cpu], 'cuda_time (us)':[cuda], 'total_time (us)':[cpu+cuda], 'count':[count] }
        df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    df_.to_csv(filename)
    et = time.perf_counter() 
    print (f"Finished Aggregating Profiles: {et - st} s")
    return     



###############################################


def replace_forward(module, ops = None): 
    _old = getattr(module, "forward")
    
    def new_forward(*args, **kwargs): 
        with torch.profiler.record_function(f"{module.__class__.__name__}_prof"): 
            return _old(*args, **kwargs)
    module_name = module.__class__.__name__.lower()
    
    if (ops is None):
        if not ( ("model" in module_name) or ("causallm" in module_name) or ("attention" in module_name) or ("decoder" in module_name) or ("mlp" in module_name) or ("sequential" in module_name) or ("block" in module_name)):
            setattr(module, "forward", new_forward)
    else: 
        if (module_name in ops): 
            setattr(module, "forward", new_forward)

@torch.no_grad()
def profile_model (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    print(f"num runs {num_prof_runs}")
    
    print(f"active {active}")
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(**input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                prof.step()
                del out
                gc.collect
    print (f"Num of Runs: {active}")
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof (prof, filename, custom)



def profile_model_dynamo (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = True, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    from torch._inductor import config
    config.cpp.enable_kernel_profile = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]="1"
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"]="1"
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"]="1"
    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof_{n}"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(**input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time for inference {n} = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof_dynamo (prof, filename, custom)



def profile_model_shape (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, 2
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = True
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes = True) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(**input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}_shape.json")
    

    filename = f"{out_dir}/{model_name}_shape.csv"
    _analyze_prof_shape (prof, filename, custom)

@torch.no_grad()
def profile_model_tv (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        st = time.perf_counter()
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
        et = time.perf_counter() 
        print (f"Exporting Trace {et - st} s")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof (prof, filename, custom)

def profile_model_dynamo_tv (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = True, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    from torch._inductor import config
    config.cpp.enable_kernel_profile = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]="1"
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"]="1"
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"]="1"
    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof_{n}"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time for inference {n} = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof_dynamo (prof, filename, custom)




def profile_model_tv_shape (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = True
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes = True) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}_shape'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        st = time.perf_counter()
        prof.export_chrome_trace(f"{out_dir}/{model_name}_shape.json")
        et = time.perf_counter() 
        print (f"Exporting Trace {et - st} s")
    

    filename = f"{out_dir}/{model_name}_shape.csv"
    _analyze_prof_shape (prof, filename, custom)

def profile_model_tv_energy(model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True):
    
    cmd = "nvidia-smi --query-gpu=index,power.draw,memory.used,utilization.memory,utilization.gpu --format=csv --loop-ms=1000 > power.log"
    process = subprocess.Popen (cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os. setsid)
    for i in range (1000): 
        out = model(input_)
        torch.cuda.synchronize()
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

def profile_model_energy(model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True):
    
    cmd = "nvidia-smi --query-gpu=index,power.draw,memory.used,utilization.memory,utilization.gpu --format=csv --loop-ms=100 > power.log"
    process = subprocess.Popen (cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os. setsid)
    for i in range (1000): 
        out = model(**input_)
        torch.cuda.synchronize()
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)



@torch.no_grad()
def profile_model_generate (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    max_num_tokens = 128,
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    num_layers = model.config.num_hidden_layers  # Number of transformer layers
    num_heads = model.config.num_attention_heads  # Number of attention heads
    head_dim = model.config.hidden_size // num_heads  # Dimension per head
    past_kv = tuple(
        (
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device),  # Random key tensor
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device)   # Random value tensor
        )
        for _ in range(num_layers)
    )
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            
            for n in range (skip_first + wait + warmup):
                out = model(**input_, )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.step()
            del out
            
            with torch.profiler.record_function(f"Inference_prof"):
                st = time.perf_counter()
                #out = model.generate(**input_, max_new_tokens = max_num_tokens, eos_token_id=None,)
                out = model(input_ids = input_.input_ids[:,:1], past_key_values = past_kv, use_cache = True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                et = time.perf_counter () - st 
                print (f"Time fot one inference = {et} s")
            
            del out
            gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof (prof, filename, custom)

def profile_model_dynamo_generate (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    max_num_tokens = 128,
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 
    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    from torch._inductor import config
    config.cpp.enable_kernel_profile = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]="1"
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"]="1"
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"]="1"
    skip_first, wait, warmup, active = 1, 2, 2, 10
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    num_layers = model.config.num_hidden_layers  # Number of transformer layers
    num_heads = model.config.num_attention_heads  # Number of attention heads
    head_dim = model.config.hidden_size // num_heads  # Dimension per head
    past_kv = tuple(
        (
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device),  # Random key tensor
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device)   # Random value tensor
        )
        for _ in range(num_layers)
    )
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            
        for n in range (skip_first + wait + warmup + 10):
            with torch.profiler.record_function(f"Inference_prof_{n}"):
                st = time.perf_counter()
                #out = model.generate(**input_, max_new_tokens = max_num_tokens, eos_token_id=None,)
                out = model(input_ids = input_.input_ids[:,:1], past_key_values = past_kv, use_cache = True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                et = time.perf_counter () - st 
                print (f"Time fot one inference = {et} s")
            prof.step()            
        del out
        gc.collect

    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof_dynamo (prof, filename, custom)

def profile_generate_shape (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    max_num_tokens = 1,
                    dynamo = False, 

                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, 2
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = True
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes = True) as prof:
            
            for n in range (skip_first + wait + warmup):
                out = model.generate (**input_, max_new_tokens = 1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.step()
            del out
            
            with torch.profiler.record_function(f"Inference_prof"):
                st = time.perf_counter()
                out = model.generate(**input_, max_new_tokens = max_num_tokens, eos_token_id=None,)
                #out = model(**input_)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                et = time.perf_counter () - st 
                print (f"Time fot one inference = {et} s")
            
            del out
            gc.collect
    
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}_shape.json")
    

    filename = f"{out_dir}/{model_name}_shape.csv"
    _analyze_prof_shape (prof, filename, custom)



################## Dataset Inputs ###################
@torch.no_grad()
def profile_model_dataset (model_name, 
                    model, 
                    input, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    print(f"num runs {num_prof_runs}")
    
    print(f"active {active}")
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for n in range (skip_first + wait + warmup + active):
                input_ = input[n].to(model.device)
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(**input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                
                del out
                del input_ 
                gc.collect
                torch.cuda.empty_cache()
                prof.step()
                
    print (f"Num of Runs: {active}")
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof (prof, filename, custom)


def profile_model_tv_dataset (model_name, 
                    model, 
                    input, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device,
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof) as prof:
            for n in range (skip_first + wait + warmup + active):
                input_ = input[n].to(device)
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                del out
                del input_ 
                gc.collect
                torch.cuda.empty_cache()
                prof.step()
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        st = time.perf_counter()
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
        et = time.perf_counter() 
        print (f"Exporting Trace {et - st} s")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof (prof, filename, custom)






def main(): 
    #profile_hf_lm_model("raed", "gpt2-large", None, False, "cuda", "./non-gemm-out", True) 
    
    #model = torchvision.models.vit_b_16().to(torch.float16)
    custom = False ## Determines Granurality of Profile, if you need coarser grain information at the torch.nn.module set to True
    #profile_model (model, "vit-b16", 'cuda', torch.randn(1,3,224,224).to(torch.float16), custom,True, './non-gemm-out')
    #profile_softmax ("softmax", "softmax", 'cuda', torch.randn(1,64).to(torch.float16), custom,True, './out')



if __name__ =="__main__": 
    main()
    #generate_report("out/vit-b16.csv")
    print ("Done")
