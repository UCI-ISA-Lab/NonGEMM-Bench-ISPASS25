import json
import argparse 
import pandas as pd
import os

gemm_ops = ['MatMul', 'Gemm', 'FusedMatMul', 'Conv', 'FusedConv', 'FusedGemm','ConvTranspose', 'eisum', 'CaskConvolution', 'gemm', 'CaskGemmConvolution', 'CaskConvolution'] 


def analyze_layers(filename):
    # Load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    # Function to split metadata string by the separator and return a set of metadata fields
    def extract_metadata_fields(metadata_str):
        if metadata_str:
            return set(metadata_str.split('\u001f'))
        return set()

    # Dictionary to hold metadata and corresponding layers
    metadata_dict = {}

    # Loop through each layer and extract metadata
    for layer in data.get('Layers', []):
        layer_name = layer.get('Name', 'Unnamed Layer')
        inputs = layer.get('Inputs', [])
        metadata = layer.get('Metadata', '')
        metadata_fields = extract_metadata_fields(metadata)
        
        # Store metadata fields for each layer
        for field in metadata_fields:
            if field not in metadata_dict:
                metadata_dict[field] = []
            metadata_dict[field].append((layer_name, inputs))

    # Print layers that share at least one common metadata field, along with their inputs
    for field, layers in metadata_dict.items():
        if len(layers) > 1:  # Only print fields that are shared by more than one layer
            print(f"Common Metadata Field: {field}")
            for layer, inputs in layers:
                print(f"Layer: {layer}")
                print("Inputs:")
                for input_info in inputs:
                    input_name = input_info.get('Name', 'Unknown Input')
                    input_dims = input_info.get('Dimensions', 'Unknown Dimensions')
                    input_format = input_info.get('Format/Datatype', 'Unknown Format')
                    print(f"  - Name: {input_name}, Dimensions: {input_dims}, Format: {input_format}")
            print("\n")
            
######   Function to classify layers         
def classify_layers_gng(layer_json: str = "", gemm_ops : list = []):
    with open(layer_json, "r") as file:
        data = json.load(file)
    
    layers = data["Layers"]
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
        for g in gemm_ops_: 
            if g in op_: 
                is_gemm = True
                gemm.append(op)
        if not (is_gemm): 
            non_gemm.append(op)
    return gemm, non_gemm, is_gemm

###############
def report_latency (profile_csv, gemm, non_gemm, model_name, input_shape, summary_csv): #input_shape is a string BatchSize_TensorDims
    df = pd.read_csv(profile_csv)
    print ("Median Analysis")
    df_median = pd.DataFrame(df[['name',' medianMs']])
    inf_total_ms = df_median[' medianMs'].sum()
    df_median[' percentage'] = df_median[' medianMs'] / inf_total_ms

    op_max = df.loc[df[' medianMs'].idxmax()]['name']
    op_min = df.loc[df[' medianMs'].idxmin()] ['name']

    df_gemm = df[df['name'].isin(gemm)]
    df_nongemm = df[df['name'].isin(non_gemm)]

    gemm_latency = df_gemm[' medianMs'].sum()
    non_gemm_latency = df_nongemm[' medianMs'].sum()
    total_latency = gemm_latency + non_gemm_latency

    new_entry = {'name':[model_name], 'gemm latency (ms)': [gemm_latency], 'non_gemm latency (ms)': [non_gemm_latency], 'total latency (ms)':[total_latency], 'input_shape':[input_shape]}

    update_summary(summary_csv, new_entry)

    print (f"Total execution time: {inf_total_ms} ms")
    # print (df_median)
    print(f"Gemm total: {df_gemm[' medianMs'].sum()} ms, Percentage: {df_gemm[' percentage'].sum()}")
    print(f"Non Gemm total: {df_nongemm[' medianMs'].sum()} ms,  Percentage: {df_nongemm[' percentage'].sum()} ")


def report_pct (profile_csv, gemm, non_gemm, model_name, input_shape, summary_csv): #input_shape is a string BatchSize_TensorDims
    df = pd.read_csv(profile_csv)
    print ("Median Analysis")
    df_median = pd.DataFrame(df[['name',' medianMs']])
    inf_total_ms = df_median[' medianMs'].sum()
    df_median[' percentage'] = df_median[' medianMs'] / inf_total_ms

    op_max = df.loc[df[' medianMs'].idxmax()]['name']
    op_min = df.loc[df[' medianMs'].idxmin()] ['name']
    # df_gemm = df[df['name'].str.contains('Gemm|MatMul|con|Fc', case = False)]
    # df_nongemm = df[~df['name'].str.contains('Gemm|MatMul|con|Fc', case = False)]
    df_gemm = df[df['name'].isin(gemm)]
    df_nongemm = df[df['name'].isin(non_gemm)]

    gemm_latency = df_gemm[' medianMs'].sum()
    non_gemm_latency = df_nongemm[' medianMs'].sum()
    total_latency = gemm_latency + non_gemm_latency

    gemm_pct = df_gemm[' percentage'].sum()
    non_gemm_pct = df_nongemm[' percentage'].sum()
    total_pct = gemm_pct + non_gemm_pct

    pct_new_entry = {'name':[model_name], 'gemm latency (%)': [gemm_pct], 'non_gemm latency (%)': [non_gemm_pct], 'input_shape':[input_shape]}
    
    update_summary(summary_csv, pct_new_entry)

    print (f"Total execution time: {inf_total_ms} ms")
    # print (df_median)
    print(f"Gemm total: {df_gemm[' medianMs'].sum()} ms, Percentage: {df_gemm[' percentage'].sum()}")
    print(f"Non Gemm total: {df_nongemm[' medianMs'].sum()} ms,  Percentage: {df_nongemm[' percentage'].sum()} ")

def update_summary(filename, new_entry): 
    file_path = filename
    columns = ['name', 'gemm latency (ms)', 'non_gemm latency (ms)', 'total latency (ms)', 'input_shape' ]  

    # Step 1: Check if the file exists
    if not os.path.exists(file_path):
        # If the file does not exist, create it with the column labels
        columns = list(new_entry.keys())
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)
        print(f"CSV file created with columns: {columns}")
    else:
        # If the file exists, read it
        df = pd.read_csv(file_path)
        print(f"CSV file already exists. Data read from {file_path}")

    # Step 2: Now you can add a new row (as needed)
    df = pd.concat((df, pd.DataFrame(new_entry)))

    # Step 3: Save the updated DataFrame back to the CSV file
    df.to_csv(file_path)


# Main function
def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Parse the TRT generated logs')

    # Add arguments
    parser.add_argument('--path_to_trt_logs', type=str, help='Path to the Directory containing TRT Logs')
    parser.add_argument('--keywords', type=str, nargs='*', default=['conv', 'matmul', 'gemm', 'fc'],
                        help='List of keywords to identify GEMM layers (space-separated)')

    parser.add_argument("--out_dir", type=str, help='Path to the output directory with the summarized data')
    parser.add_argument("--model_name", type=str, help='Model Name')
    parser.add_argument("--input_shape", type=str, help='Input Shape')
    

    # Parse arguments
    args = parser.parse_args()

    layer_json = f"{args.path_to_trt_logs}/layer.json"
    profile_csv = f"{args.path_to_trt_logs}/profile_csv.csv"

    # Classify layers
    # gemm_layers, non_gemm_layers = classify_layers(layer_json, args.keywords)
    # gemm_layers, non_gemm_layers = classify_layers_(layer_json, gemm_ops)
    gemm_layers, non_gemm_layers = classify_layers_gng(layer_json, gemm_ops)

    os.system(f"mkdir -p {args.out_dir}")
    summary_file = f"{args.out_dir}/summary.csv"
    pct_summary_file = f"{args.out_dir}/pct_summary.csv"

    report_latency(profile_csv,gemm_layers, non_gemm_layers,args.model_name, args.input_shape, summary_file)
    report_pct(profile_csv,gemm_layers, non_gemm_layers,args.model_name, args.input_shape, pct_summary_file)
    # Print the results
    print (f"Printing Results for model {args.model_name}")
    print(f"GEMM Layers: {len(gemm_layers)}")
    print(gemm_layers)
    print(f"\nNon-GEMM Layers: {len(non_gemm_layers)}")
    print(non_gemm_layers)

# Entry point for the script
if __name__ == "__main__":
    main()



