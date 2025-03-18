import os 
import subprocess
import argparse
from utils import torch_models, hf_models
from eval import eval

coco_path = "/path/to/datasets/coco"
imagenet_path = "/path/to/datasets/imagenet" 

def run_cmd (cmd, model_name):
    try: 
        status = os.system(cmd)
        # process = subprocess.run(cmd, shell = True, check = True).returncode
        # print(f"Successful Run of Model {model_name}: {process}")
        return status

    except: 
        print (f"Failed Run: Model {model_name}")
        return 1

def run_eval(model_name, hf, dataset, dataset_path,onnx_dir, backend, nb_smpls, batch_size, out_dir, task, seq_len = 512 ): 
    # try:
    x = eval(model_name, hf, dataset, dataset_path,onnx_dir, backend, nb_smpls, batch_size, out_dir, task, seq_len = seq_len )
    print(f"Successful Run of Model {model_name}")
    return x

    # except: 
    #     print (f"Failed Run: Model {model_name}")
    #     return 1
    
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
    parser.add_argument('--num_smpls', metavar ='num_smpls',  
                    required = True, dest ='num_smpls', 
                    type = int, 
                    help ='Number of Inference Runs') 
    parser.add_argument('--onnx_dir', metavar ='onnx_dir', 
                    required = True, dest ='onnx_dir', 
                    action ='append', 
                    help ='Directory containing onnx models')
    parser.add_argument('--backend', metavar ='backend', 
                    required = True, dest ='backend', 
                    action ='append', 
                    help ='target backend cpu or cuda')
    parser.add_argument('--out_dir', metavar ='out_dir', 
                    required = True, dest ='out_dir', 
                    action ='append', 
                    help ='output directory')
    args = parser.parse_args()
    hf = args.hf_model
    model_name = args.model_name 
    batch_size = args.batch_size
    num_smpls = args.num_smpls
    onnx_dir = args.onnx_dir [0]
    backend = args.backend 
    out_dir = args.out_dir[0]
    
    
############ Hugging Face Models Commands ######################
    if (hf): 
        

        # model_name = 'gpt2'
        # x = run_eval(model_name, hf, 'wikitext', None,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
        # print(f'Model Ran {model_name}: {x}')


        # model_name = 'gpt2-large'
        # x = run_eval(model_name, hf, 'wikitext', None,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
        # print(f'Model Ran {model_name}: {x}')



        model_name = 'gpt2-xl'
        x = run_eval(model_name, hf, 'wikitext', None,onnx_dir, backend, num_smpls, batch_size, out_dir, None, seq_len = 512)
        print(f'Model Ran {model_name}: {x}')


        model_name = 'llama'
        x = run_eval(model_name, hf, 'wikitext', None,onnx_dir, backend, num_smpls, batch_size, out_dir, None, seq_len = 2048 )
        print(f'Model Ran {model_name}: {x}')
        
        

        # model_name ='segformer-b0'
        # x = run_eval(model_name, hf, 'coco', coco_path, onnx_dir, backend, num_smpls, batch_size, out_dir, None )
        # print(f'Model Ran {model_name}: {x}')
        

#         # #RUN_PROF_HF_CMD =f"{python} eval.py --model_name segformer-b5 --hf_model --onnx_dir {onnx_dir} --dataset coco --dataset_path {coco_path} --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
#         # #run_cmd(RUN_PROF_HF_CMD,'segformer-b5')

#         # #RUN_PROF_HF_CMD =f"{python} eval.py --model_name detr --hf_model --onnx_dir {onnx_dir} --dataset coco --dataset_path {coco_path} --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
#         # #run_cmd(RUN_PROF_HF_CMD,'detr')
        
#         # #print(f'Model Ran Successfully: detr')


#         # #RUN_PROF_HF_CMD =f"{python} eval.py --model_name bert --hf_model --onnx_dir {onnx_dir} --dataset wikitext --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
#         # #run_cmd(RUN_PROF_HF_CMD,'bert')
        # model_name = 'bert'
        # x = run_eval(model_name, hf, 'wikitext', None,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
        # print(f'Model Ran {model_name}: {x}')


#         #RUN_PROF_HF_CMD =f"{python} eval.py --model_name maskformer --hf_model --onnx_dir {onnx_dir} --dataset coco --dataset_path {coco_path} --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
#         #run_cmd(RUN_PROF_HF_CMD,'maskformer')
        pass


# ############ Torch Models Commands ######################

#     #RUN_PROF_CMD =f"{python} eval.py --model_name swin-t  --dataset imagenet --dataset_path {imagenet_path} --onnx_dir onnx_dir --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
#     #run_cmd(RUN_PROF_CMD,'swin-t')
    # model_name ='swin-t'
    # x = run_eval(model_name, False, 'imagenet', imagenet_path,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
    # print(f'Model Ran {model_name}: {x}')
    

# #     #RUN_PROF_CMD =f"{python} eval.py --model_name swin-s  --dataset imagenet --dataset_path {imagenet_path} --onnx_dir onnx_dir --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
# #     #run_cmd(RUN_PROF_CMD,'swin-s')
    # model_name ='swin-s'
    # x = run_eval(model_name, False, 'imagenet', imagenet_path,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
    # print(f'Model Ran {model_name}: {x}')



# #     #RUN_PROF_CMD =f"{python} eval.py --model_name swin-b  --dataset imagenet --dataset_path {imagenet_path} --onnx_dir onnx_dir --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
# #     #run_cmd(RUN_PROF_CMD,'swin-b')
    # model_name ='swin-b'
    # x = run_eval(model_name, False, 'imagenet', imagenet_path,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
    # print(f'Model Ran {model_name}: {x}')

# #     # #RUN_PROF_CMD =f"{python} eval.py --model_name resnet50  --dataset imagenet --dataset_path {imagenet_path} --onnx_dir onnx_dir --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
# #     # #run_cmd(RUN_PROF_CMD,'resnet50')
# #     # model_name ='resnet50'
# #     # x = run_eval(model_name, False, 'imagenet', imagenet_path,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
# #     # print(f'Model Ran {model_name}: {x}')


# #     # #RUN_PROF_CMD =f"{python} eval.py --model_name mobilenet  --dataset imagenet --dataset_path {imagenet_path} --onnx_dir onnx_dir --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
# #     # #run_cmd(RUN_PROF_CMD,'mobilenet')
# #     # model_name ='mobilenet'
# #     # x = run_eval(model_name, False, 'imagenet', imagenet_path,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
# #     # print(f'Model Ran {model_name}: {x}')


#     # #RUN_PROF_CMD =f"{python} eval.py --model_name maskrcnn  --dataset coco --dataset_path {coco_path} --onnx_dir onnx_dir --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
#     # #run_cmd(RUN_PROF_CMD,'maskrcnn')
    # model_name ='maskrcnn'
    # x = run_eval(model_name, False, 'coco', coco_path,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
    # print(f'Model Ran {model_name}: {x}')


#     # # #RUN_PROF_CMD =f"{python} eval.py --model_name fasterrcnn  --dataset coco --dataset_path {coco_path} --onnx_dir onnx_dir --backend   {backend} --nb_smpls {num_smpls} --batch_size {batch_size} --out_dir  {out_dir} "
#     # # #run_cmd(RUN_PROF_CMD,'fasterrcnn')
    # model_name ='fasterrcnn'
    # x = run_eval(model_name, False, 'coco', coco_path,onnx_dir, backend, num_smpls, batch_size, out_dir, None )
    # print(f'Model Ran {model_name}: {x}')
    


