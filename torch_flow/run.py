import time
import torch
import transformers 
import torchvision  
import random 
import os 
import datasets as hf_datasets
from torch.utils.data import Subset
from torch.utils.data import DataLoader 
from eval import profile_model, profile_model_tv, profile_model_generate, profile_model_shape, profile_model_tv_shape, profile_model_tv_energy, profile_generate_shape, profile_model_dynamo, profile_model_dynamo_tv, profile_model_dynamo_generate, profile_model_energy, profile_model_dataset, profile_model_tv_dataset
import argparse 
import gc
torch.manual_seed(1969)
random.seed(123)

#####################################
########## Modify Here ##########
#####################################

out_dir_dynamo = "./non-gemm-dynamo"
out_dir = "./non-gemm-out"
out_dir_shapes = "./non-gemm-out-shapes"

# Number of Profiling Runs
NUM_RUNS= 50  

# Flag to Export Torch Profiler Trace
# Keep False if storage is limited  
EXPORT=False

# Path to Datasets Used 
COCO_IMAGES = "/Users/rfk/datasets/coco/images"
COCO_ANN = "/Users/rfk/datasets/coco/annotations/image_info_test2017.json"
IMAGENET_IMAGES = "/Users/rfk/datasets/imagenet/"

COCO_IMAGES = os.getenv("COCO_IMAGES")
COCO_ANN = os.getenv("COCO_IMAGES")
IMAGENET_IMAGES = os.getenv("IMAGENET_IMAGES")

# List of Operators not implemented natively in PyTorch Aten. 
# If your model uses a custom operator (user-defined torch.nn.module) that does not have an implementation in torch aten,
# you can add the name of custom operator class in this list. The Profiler will hence treat it as a leaf operator when parsing latency.
custom_ops = [ 'newgeluactivation', 'llamarmsnorm',
                'segformerdwconv', 'detrfrozenbatchnorm2d', 
                'wqlinearmmfunction', 'frozenbatchnorm2d', 
                'mistralrmsnorm','mixtralrmsnorm'] #'llamarotaryembedding', 'wqlinear_gemm',
#####################################
########## End Modify Here ##########
#####################################

def gen_random_prompt (seq_len: int = 16, batch_size: int = 1): 
    prompt_ = "random "
    prompt = ""
    for i in range (seq_len - 1): 
        prompt += prompt_ 
    if batch_size > 1: 
        prompt_list = []
        for i in range (batch_size): 
            prompt_list.append(prompt)
        return prompt_list
    return prompt



class ModelProfile: 
    def __init__(self, model = None,  model_name: str = 'gpt2' ,model_config: str = 'gpt2', preprocessor = None, device: str = 'cuda'): 
        if model is None: 
            raise ValueError("Oops! Please pass a valid instantiated model.")
        if preprocessor is None: 
            raise ValueError("What Happened! Please pass a valid input preprocessor (e.g. a tokenizer)")
        self.model = model 
        self.preprocessor = preprocessor
        self.model_name = model_name 
        self.model_config = model_config 
        self.device = device 
    
    def eval_(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, preprocessor = None, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        inputs_ = self.preprocessor(inputs)
        profile_model_tv(self.model_name, self.model, inputs_, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_shape_(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        inputs_ = self.preprocessor(inputs)
        profile_model_shape(self.model_name, self.model, inputs_, custom_ops, num_runs, self.device, False, out_dir_shapes, export)

    def eval_dynamo(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        inputs_ = self.preprocessor(inputs)

        profile_model_dynamo (self.model_name, self.model, inputs_, custom_ops, num_runs, self.device, True, out_dir_dynamo, export)
  
class LMProfile: 
    ## gpt2 
    ## gpt2-large 
    ## gpt2-xlarge 
    ## llama2
    ## llama2-awq 
    ## llama3 
    ## bert-base-uncased
    def __init__(self, model_name: str = 'gpt2' ,model_config: str = 'gpt2', device: str = 'cuda'): 

        self.model_config = model_config 
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        

        if device == "cuda":
            if ("8bit" in model_name):
                self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config).eval()
                
            else:    
                if ("Mixtral" in model_config):
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, device_map = "auto", torch_dtype = "auto").eval()
                else:
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config,).to(device).to(torch.float16).eval()
        else: 
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config).to(device).eval()
        print (self.model.dtype)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def eval_(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = self.get_wikitext_prompt(batch_size=batch_size, max_tokens=seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        print (f"num_runs {NUM_RUNS}")
        print(f"arg: {num_runs}")
        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    
    def eval_random(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        print (f"num_runs {NUM_RUNS}")
        print(f"arg: {num_runs}")
        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_shape_(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir_shapes, export)
    
    def eval_gen_(self, seq_len: int = 16, max_num_tokens: int = 16, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'gen_{self.model_name}_{self.device}_{max_num_tokens}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_generate(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, max_num_tokens, False, out_dir, export)
    
    def eval_gen_shape (self, seq_len: int = 16, max_num_tokens: int = 16, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'gen_{self.model_name}_{self.device}_{max_num_tokens}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        profile_generate_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, max_num_tokens, False, out_dir_shapes, export)
    
    def eval_dynamo(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_dynamo (self.model_name, self.model, inputs, custom_ops, num_runs, self.device, True, out_dir_dynamo, export)
    
    def eval_dynamo_gen (self, seq_len: int = 16, max_num_tokens: int = 16, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None):
        self.model_name = f'gen_{self.model_name}_{self.device}_{max_num_tokens}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        profile_model_dynamo_generate(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, max_num_tokens, False, out_dir_dynamo, export)


    def eval_energy_(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_energy(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    

    def get_wikitext_prompt(self, batch_size = 1, max_tokens=8192):
        # Load Wikitext-103 dataset
        dataset = hf_datasets.load_dataset("wikitext", "wikitext-103-v1", split="validation")

        # Concatenate text entries until we reach `max_tokens`
        token_count = 0
        prompt_text = ""

        for entry in dataset["text"]:
            if entry.strip():  # Skip empty lines
                tokens = self.tokenizer(entry, add_special_tokens=False)["input_ids"]
                if token_count + len(tokens) > max_tokens:
                    # Truncate to exactly 1024 tokens
                    needed_tokens = max_tokens - token_count
                    prompt_text += self.tokenizer.decode(tokens[:needed_tokens])
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

class SegformerProfile: 
    def __init__(self, model_name: str = 'segformer' ,model_config: str = "nvidia/segformer-b0-finetuned-ade-512-512", device: str = 'cuda'): 

        self.model_config = model_config 
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        

        if device == "cuda":
            self.model = transformers.SegformerForSemanticSegmentation.from_pretrained(self.model_config).to(device).to(torch.float16).eval()
            self.dtype = torch.float16
        else: 
            self.model = transformers.SegformerForSemanticSegmentation.from_pretrained(self.model_config).to(device).eval()
            self.dtype = torch.float
        self.processor = transformers.SegformerImageProcessor.from_pretrained(self.model_config)
    
    def eval_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 512, 512).to(self.device).to(self.dtype)}
        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export) 
    
    def eval_shape_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 512, 512).to(self.device).to(self.dtype)}
        profile_model_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir_shapes, export) 
    
    def eval_dynamo (self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        inputs = {'pixel_values':torch.randn(batch_size, 3, 512, 512).to(self.device).to(self.dtype)}
        profile_model_dynamo(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, True, out_dir_dynamo, export)

    def eval_input(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = self.get_n_samples_coco(num_runs = NUM_RUNS, batch_size=batch_size)
        profile_model_dataset(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def get_n_samples_coco(self, num_runs: int = 10, batch_size: int = 1):
        def transform(image): 
            image_ = image.resize((640,480))
            return self.processor (image_, return_tensors = 'pt')
        dataset = torchvision.datasets.CocoDetection(root= COCO_IMAGES, annFile=COCO_ANN, transform = transform)
        dataset_size = len(dataset)
        num_samples = (num_runs + 5) * batch_size
        subset_idx = random.sample(range (dataset_size), num_samples)
        subset = Subset(dataset, subset_idx,)
        del dataset 
        
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, )
        inputs_ = []
        for inp, x in dataloader:
            shape = inp['pixel_values'].shape
            shape_n = (shape[0], shape[2], shape[3], shape[4])
            shape_n_mask = (shape[0], shape[3], shape[4])
            inp['pixel_values'] = inp['pixel_values'].reshape(shape_n).to(self.model.dtype)
            #inp['pixel_mask'] = inp['pixel_mask'].reshape(shape_n_mask)
            inputs_.append(inp)
        print (f"Len Inputs {len(inputs_)}")
        
        assert len(inputs_)== num_runs+5
        del dataloader
        del subset
        gc.collect
        return inputs_


class MaskformerProfile: 
    def __init__(self, model_name: str = 'maskformer' ,model_config: str = "facebook/maskformer-swin-small-coco", device: str = 'cuda'): 

        self.model_config = model_config 
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        

        if device == "cuda":
            self.model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(self.model_config).to(device).to(torch.float).eval()
            self.dtype = torch.float
        else: 
            self.model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(self.model_config).to(device).eval()
            self.dtype = torch.float
        self.processor = transformers.MaskFormerFeatureExtractor.from_pretrained(self.model_config)
    
    def eval_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 800, 1088).to(self.device).to(self.dtype)} #, 'pixel_mask':torch.ones(batch_size, 800, 1088).to(self.device)}
        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_energy_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = inputs = {'pixel_values':torch.randn(batch_size, 3, 800, 1088).to(self.device).to(self.dtype)}
        profile_model_energy(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)

    def eval_shape_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 800, 1088).to(self.device).to(self.dtype)}
        profile_model_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir_shapes, export)
    
    def eval_input(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = self.get_n_samples_coco(num_runs = NUM_RUNS, batch_size=batch_size)
        profile_model_dataset(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def get_n_samples_coco(self, num_runs: int = 10, batch_size: int = 1):
        def transform(image): 
            image_ = image.resize((640,480))
            return self.processor (image_, return_tensors = 'pt')
        dataset = torchvision.datasets.CocoDetection(root= COCO_IMAGES, annFile=COCO_ANN, transform = transform)
        dataset_size = len(dataset)
        num_samples = (num_runs + 5) * batch_size
        subset_idx = random.sample(range (dataset_size), num_samples)
        subset = Subset(dataset, subset_idx,)
        del dataset 
        
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, )
        inputs_ = []
        for inp, x in dataloader:
            shape = inp['pixel_values'].shape
            shape_n = (shape[0], shape[2], shape[3], shape[4])
            shape_n_mask = (shape[0], shape[3], shape[4])
            inp['pixel_values'] = inp['pixel_values'].reshape(shape_n).to(self.model.dtype)
            #inp['pixel_mask'] = inp['pixel_mask'].reshape(shape_n_mask)
            inputs_.append(inp)
        print (f"Len Inputs {len(inputs_)}")
        
        assert len(inputs_)== num_runs+5
        del dataloader
        del subset
        gc.collect
        return inputs_
         
class DetrProfile: 
    def __init__(self,model_name: str = 'detr' ,model_config: str = "facebook/detr-resnet-50", device: str = 'cuda'): 

        self.model_config = model_config 
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        

        if device == "cuda":
            self.model = transformers.DetrForObjectDetection.from_pretrained(self.model_config).to(device).to(torch.float16).eval()
            self.dtype = torch.float16
        else: 
            self.model = transformers.DetrForObjectDetection.from_pretrained(self.model_config).to(device).eval()
            self.dtype = torch.float
        self.processor = transformers.DetrImageProcessor.from_pretrained(self.model_config)
    
    def eval_input(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = self.get_n_samples_coco(num_runs = NUM_RUNS, batch_size=batch_size)
        profile_model_dataset(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 800, 1088).to(self.device).to(self.dtype), 'pixel_mask':torch.ones(batch_size, 800, 1088).to(self.device)}
        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_shape_ (self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 800, 1088).to(self.device).to(self.dtype), 'pixel_mask':torch.ones(batch_size, 800, 1088).to(self.device)}
        profile_model_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir_shapes, export)

    def eval_dynamo (self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        inputs = {'pixel_values':torch.randn(batch_size, 3, 800, 1088).to(self.device).to(self.dtype), 'pixel_mask':torch.ones(batch_size, 800, 1088).to(self.device)}
        profile_model_dynamo(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, True, out_dir_dynamo, export)
    
    def get_n_samples_coco(self, num_runs: int = 10, batch_size: int = 1):
        def transform(image): 
            image_ = image.resize((640,480))
            return self.processor (image_, return_tensors = 'pt')
        dataset = torchvision.datasets.CocoDetection(root= COCO_IMAGES, annFile=COCO_ANN, transform = transform)
        dataset_size = len(dataset)
        num_samples = (num_runs + 5) * batch_size
        subset_idx = random.sample(range (dataset_size), num_samples)
        subset = Subset(dataset, subset_idx,)
        del dataset 
        
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, )
        inputs_ = []
        for inp, x in dataloader:
            shape = inp['pixel_values'].shape
            shape_n = (shape[0], shape[2], shape[3], shape[4])
            shape_n_mask = (shape[0], shape[3], shape[4])
            inp['pixel_values'] = inp['pixel_values'].reshape(shape_n).to(self.model.dtype)
            inp['pixel_mask'] = inp['pixel_mask'].reshape(shape_n_mask)
            inputs_.append(inp)
        print (f"Len Inputs {len(inputs_)}")
        
        assert len(inputs_)== num_runs+5
        del dataloader
        del subset
        gc.collect
        return inputs_

    


class HFVitProfile: 
    def __init__(self, model_name: str = 'vit-hf-base' ,model_config: str = "google/vit-base-patch16-224", device: str = 'cuda'): 
        ## "google/vit-base-patch16-224"
        ## "google/vit-large-patch16-224" 
        ## "google/vit-huge-patch14-224-in21k"

        self.model_config = model_config 
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        
        if device == "cuda":
            self.model = transformers.ViTForImageClassification.from_pretrained(self.model_config).to(device).to(torch.float16).eval()
            self.dtype = torch.float16
        else: 
            self.model = transformers.ViTForImageClassification.from_pretrained(self.model_config).to(device).eval()
            self.dtype = torch.float
        self.processor = transformers.ViTImageProcessor.from_pretrained(self.model_config)
    
    def eval_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 224, 224).to(self.device).to(self.dtype)}
        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export) 

class HFSwinProfile:
    def __init__(self, model_name: str = 'swin-hf-tiny' ,model_config: str = "microsoft/swinv2-tiny-patch4-window8-256", device: str = 'cuda'): 
        ## "microsoft/swinv2-tiny-patch4-window8-256"
        ## microsoft/swin-small-patch4-window7-224
        ## "microsoft/swinv2-base-patch4-window16-256" 
        ## "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"

        self.model_config = model_config 
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        
        if device == "cuda":
            self.model = transformers.AutoModelForImageClassification.from_pretrained(self.model_config).to(device).to(torch.float16).eval()
            self.dtype = torch.float16
        else: 
            self.model = transformers.AutoModelForImageClassification.from_pretrained(self.model_config).to(device).eval()
            self.dtype = torch.float
        self.processor = transformers.AutoImageProcessor.from_pretrained(self.model_config)
    
    def eval_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = {'pixel_values':torch.randn(batch_size, 3, 256, 256).to(self.device).to(self.dtype)}
        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)

class VitProfile: 
    variants = {
        'vit-base':torchvision.models.vit_b_16, 
        'vit-large': torchvision.models.vit_l_16,
        'vit-huge':torchvision.models.vit_h_14, 
        'swin-tiny': torchvision.models.swin_t, 
        'swin-small': torchvision.models.swin_s,
        'swin-base': torchvision.models.swin_b,
        }
    
    weights = {
        'vit-base':torchvision.models.ViT_B_16_Weights.DEFAULT, 
        'vit-large': torchvision.models.ViT_L_16_Weights.DEFAULT,
        #'l-32':[vit_l_32,ViT_L_32_Weights],
        'vit-huge': torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1, 
        'swin-tiny':torchvision.models.Swin_T_Weights.DEFAULT, 
        'swin-small': torchvision.models.Swin_S_Weights.DEFAULT,
        'swin-base':torchvision.models.Swin_B_Weights.DEFAULT,
        }

    def __init__(self, model_name: str = 'vit-huge' ,model_config: str = "google/vit-base-patch16-224", device: str = 'cuda'): 
        ## vit_b_16,ViT_B_16_Weights
        ## vit_l_16,ViT_L_16_Weights] 
        ## vit_h_14, ViT_H_14_Weights

        self.model_config = model_name  
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        variant = self.variants[self.model_name]
        weight = self.weights[self.model_name]
        if device == "cuda":
            self.model = variant().to(device).to(torch.float16).eval()
            self.dtype = torch.float16
        else: 
            self.model = variant().to(device).eval()
            self.dtype = torch.float
        self.preprocessor = self.weights[model_name].transforms()
        #self.processor = transformers.ViTImageProcessor.from_pretrained(self.model_config)
    
    def eval_input(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = self.get_n_samples_imagenet(num_runs=num_runs, batch_size=batch_size)
        profile_model_tv_dataset(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)


    def eval_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = torch.randn(batch_size, 3, 224, 224).to(self.device).to(self.dtype)
        profile_model_tv(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)

    def eval_shape_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = torch.randn(batch_size, 3, 224, 224).to(self.device).to(self.dtype)
        profile_model_tv_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir_shapes, export)

    def eval_energy_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = torch.randn(batch_size, 3, 224, 224).to(self.device).to(self.dtype)
        profile_model_tv_energy(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_dynamo(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_dynamo_{self.device}_{batch_size}'
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        inputs = torch.randn(batch_size, 3, 224, 224).to(self.device).to(self.dtype)
        profile_model_dynamo_tv(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, True, out_dir_dynamo, export)
    
    def get_n_samples_imagenet(self, num_runs: int = 10, batch_size: int = 1): 
        def transform(image): 
            image_ = image.resize((224,224))
            return self.preprocessor(image_)
        path_to_imagenet = IMAGENET_IMAGES
        model_name = self.model_name.split('_')[0]
        dataset = torchvision.datasets.ImageNet(path_to_imagenet,"val",transform= transform )
        dataset_size = len(dataset)
        num_samples = (num_runs + 5) * batch_size
        subset_idx = random.sample(range (dataset_size), num_samples)
        subset = Subset(dataset, subset_idx,)
        del dataset 
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, )
        inputs_ = []
        for inp, x in dataloader:
            inputs_.append(inp.to(self.dtype))
        print (f"Len Inputs {len(inputs_)}")
        
        assert len(inputs_)== num_runs+5
        del dataloader
        del subset
        gc.collect
        return inputs_

class RCNNPorfile: 
    variants = {
        'fasterrcnn':torchvision.models.detection.fasterrcnn_resnet50_fpn, 
        'maskrcnn': torchvision.models.detection.maskrcnn_resnet50_fpn,
        
        }
    
    weights = {
        'fasterrcnn':torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights, 
        'maskrcnn': torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights,
        }
    
    def __init__(self, model_name: str = 'fasterrcnn' ,model_config: str = "fasterrcnn", device: str = 'cuda'): 
        ## vit_b_16,ViT_B_16_Weights
        ## vit_l_16,ViT_L_16_Weights] 
        ## vit_h_14, ViT_H_14_Weights

        self.model_config = model_name  
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        variant = self.variants[self.model_name]
        weight = self.weights[self.model_name]
        if device == "cuda":
            self.model = variant(weight).to(device).to(torch.float16).eval()
            self.dtype = torch.float16
        else: 
            self.model = variant(weight).to(device).eval()
            self.dtype = torch.float
        self.device = device 
        self.preprocessor = weight.DEFAULT.transforms()
        #self.processor = transformers.ViTImageProcessor.from_pretrained(self.model_config)
    
    def eval_input(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = self.get_n_samples_coco(num_runs=num_runs, batch_size=batch_size)
        profile_model_tv_dataset(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = torch.randn(batch_size, 3, 224, 224).to(self.device).to(self.dtype)
        profile_model_tv(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)

    def eval_shape_(self, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT ):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}'
        inputs = torch.randn(batch_size, 3, 224, 224).to(self.device).to(self.dtype)
        profile_model_tv_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir_shapes, export)

    def get_n_samples_coco(self, num_runs: int = 10, batch_size: int = 1):
        def transform(image): 
            image_ = image.resize((224,224))
            return self.preprocessor(image_)
        dataset = torchvision.datasets.CocoDetection(root= COCO_IMAGES, annFile=COCO_ANN, transform = transform)
        dataset_size = len(dataset)
        num_samples = (num_runs + 5) * batch_size
        subset_idx = random.sample(range (dataset_size), num_samples)
        subset = Subset(dataset, subset_idx,)
        del dataset 
        
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, )
        inputs_ = []
        for inp, x in dataloader:
            inputs_.append(inp.to(self.dtype))
        print (f"Len Inputs {len(inputs_)}")
        
        assert len(inputs_)== num_runs+5
        del dataloader
        del subset
        gc.collect
        return inputs_

def gpt2(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape: bool = False):
    model = LMProfile('gpt2', "openai-community/gpt2", device)
    if shape: 
        model.eval_shape_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    else: 
        model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

    
def gpt2_large(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape: bool = False):
    model = LMProfile('gpt2-large', 'gpt2-large', device)
    if shape: 
        model.eval_shape_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    else: 
        model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def gpt2_xl(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape: bool = False):
    model = LMProfile('gpt2-xl', 'gpt2-xl', device)
    if shape: 
        model.eval_shape_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    else:
        model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def llama2(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-2-7b-chat-hf", shape: bool = False):
    model = LMProfile('llama2', path_weights, device)
    if shape:
        model.eval_shape_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    else:
        model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def llama2_energy(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-2-7b-chat-hf"):
    model = LMProfile('llama2', path_weights, device)
    model.eval_energy_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model


def llama2_dynamo(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-2-7b-chat-hf"):
    model = LMProfile('llama2_dynamo', path_weights, device)
    model.eval_dynamo(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def llama2_generate(seq_len: int = 8, max_num_tokens: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-2-7b-chat-hf"):
    model = LMProfile('llama2', path_weights, device)
    model.eval_gen_(seq_len, max_num_tokens , NUM_RUNS, EXPORT, custom_ops)
    del model

def llama2_shape_generate(seq_len: int = 8, max_num_tokens: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-2-7b-chat-hf"):
    model = LMProfile('llama2', path_weights, device)
    model.eval_gen_shape(seq_len, max_num_tokens , NUM_RUNS, EXPORT, custom_ops)
    del model

def llama2_dynamo_generate(seq_len: int = 8, max_num_tokens: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-2-7b-chat-hf"):
    model = LMProfile('llama2_dynamo', path_weights, device)
    model.eval_dynamo_gen(seq_len, max_num_tokens, NUM_RUNS, EXPORT, custom_ops)
    del model

def llama2_shape_shape(seq_len: int = 8, max_num_tokens: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-2-7b-chat-hf"):
    model = LMProfile('llama2', path_weights, device)
    model.eval_gen_shape(seq_len, max_num_tokens , NUM_RUNS, EXPORT, custom_ops)
    del model

def llama3(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', path_weights: str ="meta-llama/Llama-3.1-8B"):
    model = LMProfile('llama3', path_weights, device)
    model.eval_random(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def llama2_awq (seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', ):
    model = LMProfile('llama2-awq',"TheBloke/Llama-2-7B-AWQ", device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def llama3_8bit(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', ):
    model = LMProfile('llama3-8bit',"meta-llama/Llama-Guard-3-8B-INT8", device)
    model.eval_random(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def bert(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False):
    model = LMProfile('bert', 'bert-base-uncased', device)
    if shape: 
        model.eval_shape_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    else:
        model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def bert_large(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False):
    model = LMProfile('bert_large', 'bert-large-uncased', device)
    if shape: 
        model.eval_shape (seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    else:
        model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def segformer(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False):
    model = SegformerProfile(device=device)
    if shape:
        model.eval_shape_(batch_size, NUM_RUNS, EXPORT)
    else: 
        model.eval_input(batch_size, NUM_RUNS, EXPORT)

    del model

def segformer_dynamo(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'):
    model = SegformerProfile(device=device)
    model.eval_dynamo(batch_size, NUM_RUNS, EXPORT)
    del model

def segformer_b1(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'):
    model = SegformerProfile("segformer-b1","nvidia/segformer-b1-finetuned-cityscapes-1024-1024",device = device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def segformer_b3(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'):
    model = SegformerProfile("segformer-b3","nvidia/segformer-b3-finetuned-ade-512-512",device = device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def segformer_b5(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'):
    model = SegformerProfile("segformer-b5","nvidia/segformer-b5-finetuned-ade-640-640",device = device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def maskformer_small(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = MaskformerProfile(model_name = "maskformer-small", device= device )
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def maskformer_base(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False): 
    model = MaskformerProfile("maskformer-base","facebook/maskformer-swin-base-coco", device = device)
    if shape: 
        model.eval_shape_(batch_size, NUM_RUNS, EXPORT) 
    else: 
        model.eval_input(batch_size, NUM_RUNS, EXPORT)

    del model

def maskformer_base_energy(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = MaskformerProfile("maskformer-base","facebook/maskformer-swin-base-coco", device = device)
    model.eval_energy_(batch_size, NUM_RUNS, EXPORT) 
    del model

def detr_dynamo(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = DetrProfile(device = device)
    model.eval_dynamo(batch_size, NUM_RUNS, EXPORT)
    del model

def detr(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape: bool = False): 
    model = DetrProfile(device = device)
    if shape:
        model.eval_shape_(batch_size,NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_vit_base(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFVitProfile("vit-hf-base", "google/vit-base-patch14-224", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_vit_base_16(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFVitProfile("vit-hf-base", "google/vit-base-patch16-224", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_vit_large(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFVitProfile("vit-hf-large", "google/vit-large-patch16-224", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_vit_huge(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFVitProfile("vit-hf-huge", "google/vit-huge-patch14-224-in21k", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_vit_huge_16(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFVitProfile("vit-hf-huge", "google/vit-huge-patch16-224-in21k", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_swin_tiny(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFSwinProfile("swin-hf-tiny", "microsoft/swinv2-tiny-patch4-win", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_swin_base(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFSwinProfile("swin-hf-base", "microsoft/swinv2-base-patch4-window16-256", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def hf_swin_large(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = HFSwinProfile("swin-hf-large", "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft", device)
    model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def vit_base(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False): 
    model = VitProfile("vit-base", "vit-base", device)
    if shape:
        model.eval_shape(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def vit_large(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False): 
    model = VitProfile("vit-large", "vit-large", device)
    if shape:
        model.eval_shape(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def vit_huge(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool=False): 
    model = VitProfile("vit-huge", "vit-huge", device)
    if shape:
        model.eval_shape(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def vit_huge_energy(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = VitProfile("vit-huge", "vit-huge", device)
    model.eval_energy_(batch_size, NUM_RUNS, EXPORT)
    del model

def swin_tiny(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False): 
    model = VitProfile("swin-tiny", "swin-tiny", device)
    if shape:
        model.eval_shape(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def swin_small(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape: bool=False): 
    model = VitProfile("swin-small", "swin-small", device)
    if shape:
        model.eval_shape(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def swin_small_shape(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = VitProfile("swin-small", "swin-small", device)
    model.eval_shape_(batch_size, NUM_RUNS, EXPORT)
    del model

def swin_base(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False): 
    model = VitProfile("swin-base", "swin-base", device)
    if shape:
        model.eval_shape_(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_(batch_size, NUM_RUNS, EXPORT)
    del model

def swin_base_dynamo(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = VitProfile("swin-base", "swin-base", device)
    model.eval_dynamo(batch_size, NUM_RUNS, EXPORT)
    del model

def swin_tiny_dynamo(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = VitProfile("swin-tiny", "swin-tiny", device)
    model.eval_dynamo(batch_size, NUM_RUNS, EXPORT)
    del model

def fasterrcnn(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False):
    model = RCNNPorfile('fasterrcnn', 'fasterrcnn', device)
    if shape:
        model.eval_shape_(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def maskrcnn(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False): 
    model = RCNNPorfile('maskrcnn', 'maskrcnn', device)
    if shape:
        model.eval_shape_(batch_size, NUM_RUNS, EXPORT)
    else:
        model.eval_input(batch_size, NUM_RUNS, EXPORT)
    del model

def swin_small_energy(device: str = 'cuda'): 
    model = VitProfile("swin-small", "swin-small", device)
    model.eval_energy_()
    del model 

def detr_shape (seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'): 
    model = DetrProfile(device = device)
    model.eval_shape_(batch_size, NUM_RUNS, EXPORT)
    del model

def segformer_shape(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda'):
    model = SegformerProfile(device=device)
    model.eval_shape_(batch_size, NUM_RUNS, EXPORT)
    del model

def mistral_MoE(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', shape:bool = False): 
    model = LMProfile(model_name = 'mistral_MoE', model_config = "mistralai/Mixtral-8x7B-v0.1", device = device)
    if shape:
        model.eval_shape_(seq_len, batch_size, NUM_RUNS, EXPORT)
    else:
        print ("eval")
        model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT)
    del model

profiling_functions = {
    'swin-base': swin_base,
    'swin-small': swin_small,
    'swin-tiny': swin_tiny,
    'vit-huge': vit_huge,
    'vit-large': vit_large,
    'vit-base': vit_base,
    'swin-hf-large': hf_swin_large,
    'swin-hf-base': hf_swin_base,
    'swin-hf-small': hf_swin_tiny,
    'vit-hf-huge': hf_vit_huge,
    'vit-hf-large': hf_vit_large,
    'vit-hf-base': hf_vit_base,
    'detr': detr,
    'maskformer-base': maskformer_base,
    'maskformer-small': maskformer_small,
    'segformer': segformer,
    'llama2-awq': llama2_awq,
    'llama2': llama2,
    'gpt2-xl': gpt2_xl,
    'gpt2-large': gpt2_large,
    'gpt2': gpt2,
    'bert': bert, 
    'maskrcnn':maskrcnn, 
    'fasterrcnn':fasterrcnn, 
    'llama3-8bit':llama3_8bit, 
    'segformer-b1':segformer_b1,
    'segformer-b3':segformer_b3,
    'segformer-b5':segformer_b5,
    'llama3':llama3, 
    'bert_large':bert_large,
    'vit-hf-base-16':hf_vit_base_16, 
    'vit-hf-huge-16': hf_vit_huge_16,
    'mixtral':mistral_MoE,
}

def parse_arguments(): 
    parser = argparse.ArgumentParser(description ='Torch Profiling')
    
    parser.add_argument ("--model_name", dest="model_name",
                        required = True,  type = str, help = "Name of Model to profile", choices = ['swin-base', 'swin-small', 'swin-tiny', 'vit-huge', 'vit-large', 'vit-base', 'swin-hf-large', 'swin-hf-base', 'swin-hf-small', 'vit-hf-huge', 'vit-hf-large', 'vit-hf-base', 'detr', 'maskformer-base', 'maskformer-small','segformer', 'llama2-awq', 'llama2', 'gpt2-xl', 'gpt2-large', 'gpt2', 'bert', 'maskrcnn', 'fasterrcnn','segformer-b1','segformer-b3', 'llama3', 'bert_large', 'mistral', 'llama3-8bit', 'mixtral' ])
    
    parser.add_argument ("--model_weights", dest="weights",
                        required = False,  type = str, help = "Path to local weights")
    
    parser.add_argument ("--batch_size", dest="batch_size",
                        required = True,  type = int, help = "batch_size")
    
    parser.add_argument ("--seq_len", dest="seq_len",
                        required = False,  type = int, help = "Input Sequence Length for Language Models")
    
    parser.add_argument ("--device", dest="device",
                        required = True,  type = str, help = "cpu or cuda")
    
    parser.add_argument ("--out_dir", dest="out_dir",
                        required = False,  type = str, help = "Directory to store output csv files")
    
    args = parser.parse_args()
    return args

def main ():
    args = parse_arguments()
    
    model_name = args.model_name
    weight_path = args.weights if args.weights is not None else ""
    batch_size = args.batch_size
    seq_len = args.seq_len if args.seq_len is not None else 1
    device = args.device
    output_dir = args.out_dir if args.out_dir is not None else out_dir


    print(f'Profiling {model_name} on {device}')
    st = time.perf_counter()
    profiling_functions[model_name](seq_len, batch_size, device)
    et = time.perf_counter
    print (f"Finished Profiling {model_name} on {device}")

def dynamo(): 
    llama2_dynamo(seq_len = 2048)
    
    swin_base_dynamo(batch_size=1)
    swin_base_dynamo(batch_size=2)
    swin_base_dynamo(batch_size=4)
    swin_base_dynamo(batch_size=8)
    
    swin_tiny_dynamo(batch_size=1)
    swin_tiny_dynamo(batch_size=2)
    swin_tiny_dynamo(batch_size=4)
    swin_tiny_dynamo(batch_size=8)

    segformer_dynamo(batch_size=1)
    segformer_dynamo(batch_size=2)
    segformer_dynamo(batch_size=4)
    segformer_dynamo(batch_size=8)

    detr_dynamo(batch_size=1)
    detr_dynamo(batch_size=2)
    detr_dynamo(batch_size=4)
    detr_dynamo(batch_size=8)

def energy(): 
    #swin_small_energy()
    #vit_huge_energy()
    #maskformer_base_energy()
    llama2_energy(seq_len=2048)

def debug(): 
    # 

    vit_huge(batch_size = 1, device = 'cpu')
    vit_large(batch_size = 2, device = 'cpu')
    vit_base(batch_size = 4, device = 'cpu')

    swin_tiny(batch_size = 1, device = 'cpu')
    swin_small(batch_size = 2, device = 'cpu')
    swin_base(batch_size = 4, device = 'cpu')



    # segformer(batch_size = 1, device = 'cpu') 
    # segformer(batch_size = 4, device = 'cpu') 

    # maskformer_base(batch_size=1, device ="cpu")
    # maskformer_base(batch_size=2, device ="cpu")

    # detr(batch_size=1, device = 'cpu')
    # detr(batch_size=2, device = 'cpu')

    # swin_base(batch_size = 64)
    #  llama2_dynamo_generate(seq_len=2048)
    # swin_base_dynamo()
    #llama2_dynamo(seq_len = 512)
    # llama2_shape_shape(seq_len = 1, )
    # llama2_generate(seq_len = 2048, max_num_tokens=8192)
    # llama2_shape_generate(seq_len = 2048, max_num_tokens=1)
    #llama2 (2048, 1, 'cuda')
    #llama3_8bit(2048,1,'cuda')
    #model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B-INT8").eval()
    #print(model)
    #segformer_b5()
    # detr_shape()
    # detr_shape(batch_size=2)
    # detr_shape(batch_size=4)
    # detr_shape(batch_size=8)

    # segformer_shape(batch_size=1)
    # segformer_shape(batch_size=2)
    # segformer_shape(batch_size=4)
    # segformer_shape(batch_size=8)

    # llama3(seq_len = 128, device = 'cpu')
    # llama3(seq_len = 128, device = 'cuda')
    # llama3(seq_len = 512, device = 'cpu')
    # llama3(seq_len = 512, device = 'cuda')
    # llama3(seq_len = 1024, device = 'cpu')
    # llama3(seq_len = 1024, device = 'cuda')

    # bert(seq_len = 128, device = 'cpu')
    # bert(seq_len = 128, device = 'cuda')

    # bert_large(seq_len = 128, device = 'cpu')
    # bert_large(seq_len = 128, device = 'cuda')

    # hf_vit_base_16(device = 'cpu')
    # hf_vit_base_16(device='cuda')
    

    # hf_vit_huge(device = 'cpu')
    # hf_vit_huge(device='cuda')

    # mistral_MoE(seq_len=2048, device = 'cuda')

    #swin_small_shape()
    







    pass 

if __name__ == "__main__": 
    main()
    pass
