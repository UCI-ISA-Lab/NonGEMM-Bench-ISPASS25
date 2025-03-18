import os 
import json 
import onnx 
import argparse
import pandas as pd 
import onnx_graphsurgeon as surgeon



gemm_ops = ['MatMul', 'Gemm', 'FusedMatMul', 'Conv', 'FusedConv', 'FusedGemm','ConvTranspose', 'eisum', 'CaskConvolution', 'gemm', 'CaskGemmConvolution'] 
gemm_ops = [op.lower() for op in gemm_ops]

def get_all_metada(layer_json):
    with open(layer_json, "r") as file:
        data = json.load(file)

    layers = data['Layers'] 
    old_layers = []
    meta = 0 
    for layer in layers: 
        if ("Metadata" in list(layer.keys())):
            metadata = layer["Metadata"]
            if (metadata != ""):
                result, old_ = split_metadata(metadata)
                old_ = [o.replace("ONNX Layer: ", "") for o in old_]
                old_layers = old_layers + old_ if not (old_ == "") else old_layers
                meta += len(result)
    return old_layers


def split_metadata(metadata): 
    tmp = metadata.replace("[","").replace("]","").split("\u001f")
    if len(tmp) == 1:
        tmp = tmp[0].split("\u001e")
    tmp = [t.replace('\x1e',"','") for t in tmp]
    result = [layer.split('/')[-1] for layer in tmp] 
    return result, tmp

def check_if_gemm_fused(fused_operator, gemm_ops): 
    gemm_ops_ = [x.lower() for x in gemm_ops]
    is_gemm = False
    gemm = []
    non_gemm = []
    for op in fused_operator: 
        op_ = op.lower() 
        flag = False
        for g in gemm_ops_: 
            if g in op_: 
                is_gemm = True
                gemm.append(op)
                flag = True
        if not (flag): 
            non_gemm.append(op)
    assert (len(gemm) + len(non_gemm)) == len(fused_operator)
    return gemm, non_gemm, is_gemm

def classify_layers_gng(layer_json: str = "", gemm_ops : list = []):
    with open(layer_json, "r") as file:
        data = json.load(file)

    layers = data["layers"]
    gemm = []
    non_gemm = []
    for layer in layers: 
        layer_name = layer["Name"]
        layer_type = layer["LayerType"]
        metadata = layer["Metadata"]
        if (metadata == ""):
            if (layer_type in gemm_ops): 
                gemm.append(layer_name)
            else: 
                non_gemm.append(layer_name)     
        else:
            fused_operator, tmp = split_metadata(metadata)
            x, y, is_gemm = check_if_gemm_fused(fused_operator, gemm_ops)
            if (is_gemm): 
                gemm.append(layer_name)
            else: 
                non_gemm.append(layer_name)
         
    return gemm, non_gemm 

def analyze_non_gemm_fusion (layer_json: str = "", gemm_ops : list = []): 
    # get non-gemm operators from the original graph that are fused with gemm 
    # get non-gemm operators from the original graph that are fused with non-gemm 
    with open(layer_json, "r") as file:
        data = json.load(file)
    layers = data["Layers"]
    with_gemm = []
    with_non_gemm = []
    total = []
    for layer in layers: 
        layer_name = layer["Name"]
        layer_type = layer["LayerType"].lower()
        metadata = layer["Metadata"]
        fused_operator, tmp = split_metadata(metadata)
        if (len(fused_operator) > 1):
            x, non_gemm, is_gemm = check_if_gemm_fused(fused_operator, gemm_ops)
            if (is_gemm): 
                with_gemm = with_gemm + non_gemm
            else: 
                with_non_gemm = with_non_gemm + non_gemm
    total = with_gemm + with_non_gemm

    return total, with_gemm, with_non_gemm
        

def get_dropped_nodes (onnx_file, layer_json,): 
    graph = surgeon.import_onnx(onnx.load(onnx_file))
    pre_layers = get_all_metada(layer_json)
    
    dropped = []
    kept = []
    for node in graph.nodes: 
        
        if (node.name in pre_layers): 
            kept.append(node) 
        else: 
            dropped.append(node)

    return dropped, kept

def analyze_fusion_rate(layer_json: str = './layer.json', onnx_file: str = ""): 
    
    if (onnx_file == ""): 
        print (f"Please Enter the path to the desired ONNX Model. Not Analyzing Fusion Rate")
        exit()

    graph = surgeon.import_onnx(onnx.load(onnx_file)) 
    num_og_gemm, num_og_non_gemm = get_gng_num(graph.nodes, gemm_ops)

    total, gemm, non_gemm = analyze_non_gemm_fusion(layer_json, gemm_ops)
    num_total_fused_ops = len(total)
    num_fused_gemm_ops = len(gemm)
    num_fused_non_gemm_ops = len(non_gemm)

    dropped, kept = get_dropped_nodes(onnx_file,layer_json)

    num_og_nodes = len(graph.nodes)
    num_dropped_nodes = len(dropped)
    num_kept_nodes = len(kept)
    num_pre_fusion_nodes = num_og_nodes - num_dropped_nodes
    num_kept_gemm_nodes, num_kept_non_gemm_nodes = get_gng_num(kept, gemm_ops)



    print ("\n")
    print ("Before TRT")
    print (f"Number of Nodes in ONNX Graph before applying TRT: {num_og_nodes}, ({num_kept_nodes + num_dropped_nodes})")
    print (f"The number of NonGEMM nodes in OG graph: {num_og_non_gemm}")
    print (f"The number of GEMM nodes in OG graph: {num_og_gemm}")

    print("\n")
    print ("TRT Fusion Applied")
    print (f"Total Number of NonGEMM Operators fused {num_total_fused_ops}")
    print (f"Total Number of Fused GEMM Ops {num_fused_gemm_ops}")
    print (f"Total Number of Fused NonGEMM Ops {num_fused_non_gemm_ops}")


    print ("\n")

    print (f"Number of Nodes from ONNX Graph before applying TRT: {num_og_nodes}")
    print (f"Number of Nodes Fused from original ONNX Graph: {num_pre_fusion_nodes}")
    print (f"The number of NonGEMM nodes after Dropping nodes: {num_kept_non_gemm_nodes}")
    print (f"The number of nodes dropped (no transformation applied): {len(dropped)}")

    fusion_rate = num_total_fused_ops / num_kept_nodes * 100
    gemm_fusion_rate = num_fused_gemm_ops / num_kept_nodes * 100 
    non_gemm_fusion_rate = num_fused_non_gemm_ops / num_kept_nodes * 100 

    print ("\n")
    print ("Fusion Rate (With Respect to Kept # Nodes)")
    print (f"Fusion Rate is: {num_total_fused_ops} / {num_kept_nodes} = {fusion_rate} %.")
    print (f"GEMM Fusion Rate w.t.r. to all nodes is: {gemm_fusion_rate} %.")
    print (f"NonGEMM Fusion Rate w.t.r. to all nodes is: {non_gemm_fusion_rate} %.")

    
    #fusion_rate = num_total_fused_ops / num_kept_non_gemm_nodes * 100
    gemm_fusion_rate = num_fused_gemm_ops / num_kept_gemm_nodes * 100 
    non_gemm_fusion_rate = num_fused_non_gemm_ops / num_kept_non_gemm_nodes * 100 
    print ("Fusion Rate With Respect to NonGEMM Nodes")
    print (f"GEMM Fusion Rate w.t.r. to GEMM nodes is: {num_fused_gemm_ops} / {num_kept_gemm_nodes} = {gemm_fusion_rate} %.")
    print (f"NonGEMM Fusion Rate w.t.r. to NonGEMM nodes is: {num_fused_non_gemm_ops} / {num_kept_non_gemm_nodes} = {non_gemm_fusion_rate} %.")


    fusion_rate = num_total_fused_ops / num_og_nodes * 100
    gemm_fusion_rate = num_fused_gemm_ops / num_og_nodes * 100 
    non_gemm_fusion_rate = num_fused_non_gemm_ops / num_og_nodes * 100 

    print ("\n")
    print ("\n")
    print ("Fusion Rate (With Respect to Original # Nodes)")
    print (f"Fusion Rate is: {num_total_fused_ops} / {num_og_nodes} = {fusion_rate} %.")
    print (f"GEMM Fusion Rate w.t.r. to all nodes is: {gemm_fusion_rate} %.")
    print (f"NonGEMM Fusion Rate w.t.r. to all nodes is: {non_gemm_fusion_rate} %.")

    
    #fusion_rate = num_total_fused_ops / num_kept_non_gemm_nodes * 100
    gemm_fusion_rate = num_fused_gemm_ops / num_og_gemm * 100 
    non_gemm_fusion_rate = num_fused_non_gemm_ops / num_og_non_gemm * 100 
    print ("Fusion Rate With Respect to NonGEMM Nodes")
    print (f"GEMM Fusion Rate w.t.r. to GEMM nodes is: {num_fused_gemm_ops} / {num_og_gemm} = {gemm_fusion_rate} %.")
    print (f"NonGEMM Fusion Rate w.t.r. to NonGEMM nodes is: {num_fused_non_gemm_ops} / {num_og_non_gemm} = {non_gemm_fusion_rate} %.")

    
def get_dropped_types(dropped):
    dropped_types = [n.op for n in dropped]
    dropped_types = list(set(dropped_types))
    name_dropped_per_type = {}
    num_dropped_per_type = {}
    for type in dropped_types: 
        x = 0
        tmp = []
        for n in dropped: 
            if n.op == type: 
                x+=1
                tmp.append(n)
        name_dropped_per_type [type] = tmp 
        num_dropped_per_type [type] = x 

    print (dropped_types)
    print (num_dropped_per_type)

def get_gng_num(nodes, gemm_ops): 
    non_gemm = 0
    gemm = 0 
    for node in nodes: 
        if (node.op.lower() in gemm_ops): 
            gemm +=1
        else: 
            non_gemm +=1 
    assert (gemm + non_gemm) == len(nodes)
    return gemm, non_gemm


def main(): 
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Analyze the Fusion Rate of TRT')
    
    parser.add_argument('--path_to_trt_logs', type=str, help='Path to the Directory containing layer.json in TRT Logs')
    parser.add_argument('--onnx_file', type=str,)
    
    args = parser.parse_args()

    onnx_file = args.onnx_file 
    layer_json = f"{args.path_to_trt_logs}/layer.json"
    analyze_fusion_rate(layer_json, onnx_file)

if __name__ == "__main__": 
    main()