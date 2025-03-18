import torch 
import torchvision
import transformers 
from transformers import (AutoModelForCausalLM, AutoModelForSemanticSegmentation, AutoModelForObjectDetection, BertModel, AutoModelForImageClassification, AutoTokenizer, AutoFeatureExtractor)
import os 
import argparse
import utils as u 

""" path_llama = "/Users/rfk/llama/llama/llama-2-7b-hf/" 
hf_models = {"gpt2":("gpt2",'text-generation-with-past'), "gpt2-large":("gpt2-large",'text-generation-with-past'), "gpt2-xl":("gpt2-xl",'text-generation-with-past'), 'llama':(path_llama, 'text-generation-with-past'), 'segformer-b0':('nvidia/segformer-b0-finetuned-ade-512-512', 'semantic-segmentation'), 'detr':('facebook/detr-resnet-50', 'object-detection'), 'bert':('bert-base-uncased', ) }




hf_configs = ['gpt2', 'gpt2-large', 'gpt2-xl', 'llama', 'detr', 'nvidia/segformer-b0-finetuned-ade-512-512', 'nvidia/segformer-b5-finetuned-ade-512-512', 'detr', 'bert-base-uncased', ]

cmd = f"optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/"

def convert_hf_to_onnx (model_list = None, dev = 'cpu' ): 
    if (model_list is not None):
        models = model_list
    else: 
        models = hf_configs
    for key,value in models.items(): 
        print (f'Model Being Exported: {key}')
        out_dir = f"onnx/hf/test/{key}_onnx" 
        model = value[0]
        task = value [1] 
        batch_size = value[2]
        os.system(f"mkdir -p {out_dir}")
        os.system(f"optimum-cli export onnx --model {model} --task {task} --batch_size {batch_size} --device {dev} {out_dir}")
        print ("Model Exported!!") """



if __name__ =="__main__": 
    parser = argparse.ArgumentParser(description ='ONNX Configurations')
    
    parser.add_argument('--model_name', dest ='model_name', 
                        metavar ='model_name', nargs ='*', required=True,
                        help = "Model Name to access model registry")  
    parser.add_argument('--hf_model', dest ='hf_model', 
                    required = False,  
                    action ='store_true', 
                    help ='Hugging Face model using transformers library')

    parser.add_argument('--backend', metavar ='backend',
                    required = True, dest ='backend', 
                    type = str, 
                    help ='Target Backends: cpu or cuda')
    
    parser.add_argument('--dtype', metavar ='dtype',
                    required = False, dest ='dtype', 
                    type = str,
                    default = "float", 
                    help ='Model Precision')
    
    parser.add_argument('--batch_size', metavar ='batch_size',  
                    required = True, dest ='batch_size', 
                    type = int, 
                    help ='Batch Size')
    parser.add_argument('--out_dir', metavar ='out_dir', 
                    required = True, dest ='out_dir', 
                    type = str, 
                    help ='Output Directory to store .onnx models')
    args = parser.parse_args()
    hf = args.hf_model
    model_name = args.model_name 
    backend = args.backend 
    out_dir = args.out_dir
    batch_size = args.batch_size 

    
    
    if (hf): 
        if (model_name[0] == "all"): 
            model_name = list(u.hf_models.keys())

        u.convert_hf_to_onnx(model_name, out_dir, batch_size, dev= backend)
    else: 
        if (model_name[0] == "all"): 
            model_name = list(u.torch_models.keys())
        u.convert_models_onnx(model_name, out_dir)
        
    