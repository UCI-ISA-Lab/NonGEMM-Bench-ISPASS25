import os 
import argparse
python = 'python'
batch_size = 1


EXPORT_HF_CMD = f"{python} export_onnx.py --model_name all --hf_model --backend cuda --batch_size {batch_size} --out_dir onnx_dir" 

EXPORT_TORCH_CMD = f"{python} export_onnx.py --model_name all  --backend cuda --batch_size {batch_size} --out_dir onnx_dir" 



if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description ='ONNX Configurations')
    
    parser.add_argument('--model_name', dest ='model_name', 
                        metavar ='model_name', nargs ='*', required=True,
                        help = "Model Name to access model registry")  
    parser.add_argument('--hf_model', dest ='hf_model', 
                    required = False,  
                    action ='store_true', 
                    help ='Hugging Face model using transformers library')
    parser.add_argument('--batch_size', metavar ='batch_size',  
                    required = True, dest ='batch_size', 
                    type = int, 
                    help ='Batch Size') 
    
    args = parser.parse_args()
    hf = args.hf_model
    model_name = args.model_name 
    batch_size = args.batch_size 

    
    x=''
    for i in model_name: 
        x =f'{x} {i}'
    
    if hf:
        EXPORT_HF_CMD = f"{python} export_onnx.py --model_name {x} --hf_model --backend cuda --batch_size {batch_size} --dtype float16 --out_dir onnx" 
        os.system(EXPORT_HF_CMD)
    else: 
        EXPORT_TORCH_CMD = f"{python} export_onnx.py --model_name {x}  --backend cuda --batch_size {batch_size} --dtype float16 --out_dir onnx" 
        os.system(EXPORT_TORCH_CMD)
