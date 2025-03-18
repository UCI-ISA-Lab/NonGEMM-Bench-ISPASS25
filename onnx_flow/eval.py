import argparse
import utils as u
import random 
import datasets 
import gc
import os 
from torch.utils.data import Subset, DataLoader
seed = 123 
random.seed(seed)
nb_profiles = 128
NUM_RUNS = os.environ("NUM_SMPLS")

def eval(model_name, hf, dataset, dataset_path,onnx_dir, backend, nb_smpls, batch_size, out_dir, task, seq_len = 512 ):
    hf = hf
    model_name = model_name
    onnx_dir = onnx_dir #f onnx_dir != None else onnx_dir
    backend = backend #[0]
    out_dir = out_dir #[0] if args.out_dir != None else out_dir
    dataset = dataset #[0] if args.dataset_name != None else dataset
    dataset_path = dataset_path #[0] if args.dataset_path != None else dataset_path
    nb_smpls = nb_smpls 
    batch_size = batch_size 
    task = task #[0] if args.task != None else args.task
    
    """ print (hf)
    print (model_name)
    print (onnx_dir)
    print (backend)
    print (out_dir)
    print (dataset)
    print (dataset_path)
    print (nb_smpls)
    print (batch_size)
    print (task) """

    # try:
    if (not hf):
        task = 'cv'
        preproc = u.torch_models[model_name][-1]
        model_onnx = f"{onnx_dir}/{model_name}.onnx"
        model_config = None
    else: 
        task = u.hf_models[model_name][1]
        if (not ('text' in task)):
            task = 'cv'
        
        model_onnx = u.hf_models [model_name][-1]
        model_config = f"{onnx_dir}/{model_name}_onnx"
        preproc = u.hf_models[model_name][-2].from_pretrained (model_config)
        
    x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)

    ## TODO: Move Dataset to utils, Create an interface
    if (dataset == 'wikitext'): 
        input_data = get_wikitext_prompt(x.preprocessor, batch_size=batch_size, max_tokens = seq_len)
        for i in range (NUM_RUNS):
            x.run_profile(input_data)
        x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
        return
    ##############################
    ########## UNSTABLE ##########
    ##############################
    elif (dataset == 'coco'):
        path = f'{dataset_path}/images'
        path_ann = f'{dataset_path}/annotations/image_info_test2017.json'
        
        coco = u.Coco(path, path_ann, preproc)
        dataloader = coco.set_up_dataloader(batch_size, nb_smpls) 
        print (batch_size)
        print (nb_smpls)
        run = 0
        for batch in dataloader: 
            if (run >= nb_profiles): 
                x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
                run = 0
                del x
                x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)
            if (hf): 
                #print (f"Debugging: {len(batch[1])}")
                #print (f"Debugging: {len(batch[0]['pixel_values'])}")
                #print (f"Debugging: {(batch[0]['pixel_values'][0].shape)}")
                batch[0]['pixel_values']= batch[0]['pixel_values'][0]
            #print (f"Debugging: {batch[0]}")
            
            x.run_profile(batch[0])
            run += 1
        x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')    
    elif (dataset == 'imagenet'): 
        path = f'{dataset_path}'
        imagenet = u.Imagenet(path, preproc)
        dataloader = imagenet.set_up_dataloader(batch_size, nb_smpls) 
        run = 0
        for batch in dataloader: 
            if (run > nb_profiles): 
                x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
                run = 0
                del x
                x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)
            if (hf): 
                batch[0]['pixel_values']= batch[0]['pixel_values'].squeeze(1)
            x.run_profile(batch[0])
            run += 1
        x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
    ##################################
    ########## END UNSTABLE ##########
    ##################################
    else:
        print ("Support for more custom Datasets to be implemented!")
        pass
    return 0


def get_wikitext_prompt( tokenizer, batch_size = 1, max_tokens=8192):
    # Load Wikitext-103 dataset
    dataset = datasets.load_dataset("wikitext", "wikitext-103-v1", split="validation")

    # Concatenate text entries until we reach `max_tokens`
    token_count = 0
    prompt_text = ""

    for entry in dataset["text"]:
        if entry.strip():  # Skip empty lines
            tokens = tokenizer(entry)["input_ids"]
            if token_count + len(tokens) > max_tokens:
                # Truncate to exactly Needed tokens
                needed_tokens = max_tokens - token_count
                prompt_text += tokenizer.decode(tokens[:needed_tokens])
                token_count += needed_tokens
                break
            else:
                prompt_text += entry + " "
                token_count += len(tokens)
    prompt_ = prompt_text.strip()  
    prompt_batched = []
    for _ in range(batch_size): 
        prompt_batched.append(prompt_)

    del dataset
    gc.collect()
    return prompt_batched    
    



if __name__ == "__main__": 
    ##############################
    ########## UNSTABLE ##########
    ##############################
    parser = argparse.ArgumentParser(description ='ONNX Configurations')
    
    parser.add_argument('--model_name', dest ='model_name', 
                        metavar ='model_name', nargs ='*', required=True,
                        help = "Model Name to access model registry")
    
    parser.add_argument('--hf_model', dest ='hf_model', 
                    required = False,  
                    action ='store_true', 
                    help ='Hugging Face model using transformers library')
    
    parser.add_argument('--dataset', metavar ='dataset_name', 
                    required = False, dest ='dataset_name', 
                    action ='append', 
                    help ="Dataset Name: 'wikitext', 'imagenet','coco'")
    
    parser.add_argument('--dataset_path', metavar ='dataset_path', 
                    required = False, dest ='dataset_path', 
                    action ='append', 
                    help ='Path to dataset for imagenet, coco, and custom dataset')
    
    parser.add_argument('--onnx_dir', metavar ='onnx_dir', 
                    required = True, dest ='onnx_dir', 
                    action ='append', 
                    help ='Directory containing onnx models')
    parser.add_argument('--backend', metavar ='backend', nargs ='*',
                    required = True, dest ='backend', 
                    action ='append', 
                    help ='Target Backends: [cpu] or [cuda] ')
    
    parser.add_argument('--nb_smpls', metavar ='nb_smpls', 
                    required = True, dest ='nb_smpls', type= int, 
                    help ='Number of Data Samples to be fetched from dataset')
    parser.add_argument('--batch_size', metavar ='batch_size', 
                    required = True, dest ='batch_size', 
                    type= int,
                    help ='Batch Size')
    parser.add_argument('--out_dir', metavar ='out_dir', 
                    required = True, dest ='out_dir', 
                    action ='append', 
                    help ='Output Directory to store collected profiling data')
    parser.add_argument('--task', metavar ='task', 
                    required = False, dest ='task', 
                    action ='append', 
                    help ='Output Directory to store collected profiling data')
    args = parser.parse_args()
    hf = args.hf_model
    model_name = args.model_name
    onnx_dir = args.onnx_dir[0] if args.onnx_dir != None else args.onnx_dir
    backend = args.backend [0]
    out_dir = args.out_dir[0] if args.out_dir != None else args.out_dir
    dataset = args.dataset_name [0] if args.dataset_name != None else args.dataset_name
    dataset_path = args.dataset_path [0] if args.dataset_path != None else args.dataset_path
    nb_smpls = args.nb_smpls 
    batch_size = args.batch_size 
    task = args.task[0] if args.task != None else args.task
    

    model_name = model_name[0]


    if (not hf):
        task = 'cv'
        preproc = u.torch_models[model_name][-1]
        model_onnx = f"{onnx_dir}/{model_name}.onnx"
        model_config = None
    else: 
        task = u.hf_models[model_name][1]
        if (not ('text' in task)):
            task = 'cv'
        
        model_onnx = u.hf_models [model_name][-1]
        model_config = f"{onnx_dir}/{model_name}_onnx"
        preproc = u.hf_models[model_name][-2].from_pretrained (model_config)
        
    x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)

    ## TODO: Move Dataset to utils, Create an interface
    if (dataset == 'wikitext'): 
        wiki = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
        wiki = wiki.to_pandas()
        wiki = datasets.Dataset.from_pandas(wiki[wiki['text']!=""])
        subset_idx = random.sample(range(len(wiki)), nb_smpls)
        subset = Subset(wiki, subset_idx)
        run = 1
        x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)
        num_batches = nb_smpls // batch_size 
        first_batch_size = len(subset) % batch_size 
        if (first_batch_size == 0): 
            first_batch_size = batch_size
        input_data = subset[:first_batch_size]['text']
        x.run_profile(input_data)
        for batch_num in range (1, num_batches, 1):
            if (run > nb_profiles): 
                x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
                run = 0
                del x
                x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)
            input_data = subset[first_batch_size + ((batch_num - 1) * batch_size)  :  first_batch_size + ((batch_num) * batch_size) ]['text']
            x.run_profile(input_data)
            run += 1
        x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
    elif (dataset == 'coco'):
        path = f'{dataset_path}/images'
        path_ann = f'{dataset_path}/annotations/image_info_test2017.json'
        
        coco = u.Coco(path, path_ann, preproc)
        dataloader = coco.set_up_dataloader(batch_size, nb_smpls) 
        print (batch_size)
        print (nb_smpls)
        run = 0
        for batch in dataloader: 
            if (run > nb_profiles): 
                x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
                run = 0
                del x
                x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)
            if (hf): 
                #print (f"Debugging: {len(batch[1])}")
                #print (f"Debugging: {len(batch[0]['pixel_values'])}")
                #print (f"Debugging: {(batch[0]['pixel_values'][0].shape)}")
                batch[0]['pixel_values']= batch[0]['pixel_values'][0]
            #print (f"Debugging: {batch[0]}")
            
            x.run_profile(batch[0])
            run += 1
        x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')    
    elif (dataset == 'imagenet'): 
        path = f'{dataset_path}'
        imagenet = u.Imagenet(path, preproc)
        dataloader = imagenet.set_up_dataloader(batch_size, nb_smpls) 
        run = 0
        for batch in dataloader: 
            if (run > nb_profiles): 
                x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
                run = 0
                del x
                x = u.reset (model_onnx, model_config, model_name, hf, task , backend, preproc)
            if (hf): 
                batch[0]['pixel_values']= batch[0]['pixel_values'].squeeze(1)
            x.run_profile(batch[0])
            run += 1
        x.postprocess(f'{out_dir}/{model_name}/batch_size{batch_size}')
    else:
        print ("Support for more custom Datasets to be implemented!")
        pass
    
    ##################################
    ########## END UNSTABLE ##########
    ##################################
    