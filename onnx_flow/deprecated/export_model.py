import os 
import argparse
python = 'python'
batch_size = 1


EXPORT_HF_CMD = f"{python} export_onnx.py --model_name all --hf_model --backend cuda --batch_size {batch_size} --out_dir onnx_dir" 

EXPORT_TORCH_CMD = f"{python} export_onnx.py --model_name all  --backend cuda --batch_size {batch_size} --out_dir onnx_dir" 


def export_llm_onnx (model_name, model_path, _out_dir, device, optimize): 
    print (f"########## Exporting to ONNX ###########\n\n")

    out_dir = f"onnx/{model_name}_onnx" if _out_dir is None else _out_dir
    optimization = f"--fp16 "
    x = os.system(f"mkdir -p {out_dir}")
    
    x = os.system(f"optimum-cli export onnx --model {model_path} --task text-generation-with-past --device {device} {optimization} {out_dir}")
    
    print ("########## Model Exported!! ###########\n\n")

def export_segformer_onnx (model_name, model_path, _out_dir, device, optimize): 
    print (f"########## Exporting to ONNX ###########\n\n")
    out_dir = f"onnx/{model_name}_onnx" if _out_dir is None else _out_dir
    precision = f"--dtype fp16 "
    optimization = f"--optimize O4" if optimize else ""
    x = os.system(f"mkdir -p {out_dir}")
    
    x = os.system(f"optimum-cli export onnx --model {model_path} --task image-segmentation --device {device} {precision}  {optimization} {out_dir}")
    
    print ("########## Model Exported!! ###########\n\n")

def export_DETR_onnx (model_name, model_path, _out_dir, device, optimize): 
    print (f"########## Exporting {model_name}  to ONNX ###########\n\n")
    out_dir = f"onnx/{model_name}_onnx" if _out_dir is None else _out_dir
    optimization = f"--fp16 "
    x = os.system(f"mkdir -p {out_dir}")
    
    x = os.system(f"optimum-cli export onnx --model {model_path} --task object-detection --device {device} {optimization} {out_dir}")
    
    print (f"########## Model {model_name} Exported!! ###########\n\n")

def export2onnx(): 
    path_llama=""
    # export_llm_onnx('gpt2', 'openai-community/gpt2',None, 'cuda', False )
    # export_llm_onnx('gpt2-large', 'gpt2-large', None, 'cuda',False)
    export_llm_onnx('gpt2-xl', 'openai-community/gpt2-xl', None, 'cuda',False)
    export_llm_onnx('llama2', 'meta-llama/Llama-2-7b-chat-hf', None, 'cuda',False)
    export_segformer_onnx('segformer', 'nvidia/segformer-b0-finetuned-ade-512-512', None, 'cuda',False)
    export_DETR_onnx('detr', 'facebook/detr-resnet-50', None, 'cuda',False)

    # # export_llm_onnx('gpt2_opt', 'openai-community/gpt2',None, 'cuda', True )
    # # export_llm_onnx('gpt2-large_opt', 'gpt2-large', None, 'cuda',True)
    # # export_llm_onnx('gpt2-xl_opt', 'openai-community/gpt2-xl', None, 'cuda',True)
    # #export_llm_onnx('llama2_opt', "meta-llama/Llama-2-7b-chat-hf", None, 'cuda',True)
    # export_segformer_onnx('segformer_opt', 'nvidia/segformer-b0-finetuned-ade-512-512', None, 'cuda',True)
    # export_DETR_onnx('detr_opt', 'facebook/detr-resnet-50', None, 'cuda',True)


    

if __name__ =='__main__': 
    
    export2onnx()
    # parser = argparse.ArgumentParser(description ='ONNX Configurations')
    
    # parser.add_argument('--model_name', dest ='model_name', 
    #                     metavar ='model_name', nargs ='*', required=True,
    #                     help = "Model Name to access model registry")  
    # parser.add_argument('--hf_model', dest ='hf_model', 
    #                 required = False,  
    #                 action ='store_true', 
    #                 help ='Hugging Face model using transformers library')
    # parser.add_argument('--batch_size', metavar ='batch_size',  
    #                 required = True, dest ='batch_size', 
    #                 type = int, 
    #                 help ='Batch Size') 
    
    # args = parser.parse_args()
    # hf = args.hf_model
    # model_name = args.model_name 
    # batch_size = args.batch_size 