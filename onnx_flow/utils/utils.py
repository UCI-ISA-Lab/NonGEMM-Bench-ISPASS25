import torch 
import transformers 
import optimum as op
import onnxruntime as ort 
import datasets
from torchvision import datasets as vision_data
from datasets import load_dataset
from torch.utils.data import Subset, DataLoader, Dataset
import random
import numpy as np
import os
seed = 34
random.seed(seed)

class ProfileONNX: 
    def set_InferenceSess_Options (self, model_name, backend = ["cuda"]):
        self.options = ort.SessionOptions()
        self.options.profile_file_prefix = f'{model_name}_'
        self.options.enable_profiling = True
        self.options.log_severity_level = 0
        if (len (backend) == 1):
            if (backend[0]=="cpu"): 
                self.providers = ["CPUExecutionProvider"]
                self.backend = 'cpu'
            elif (backend[0]=="cuda" or backend [0]=="gpu"):
                if (torch.cuda.is_available()): 
                    self.providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),'CPUExecutionProvider']
                    #self.providers = "CUDAExecutionProvider"
                    self.backend = 'cuda'
                else: 
                    print("CUDA Not Supported!!")
                    print ("Executing on CPU only")
                    self.providers = ["CPUExecutionProvider"]
                    self.backend = 'cpu'

            else: 
                print (f"{backend} Not Supported!!")
                exit() 
        else: 
            print (f"Backend {backend} not Implemented")
            exit()
    
    def __init__(self, model_onnx = None, model_name="model", hf = False, task = None, backend= ["cuda"], preproc = None):
        self.model = model_onnx # onnx graph for general torch models, ortModel for HF models
        self.model_name = model_name 
        self.preproc = None
        if (hf):
            self.task = task 
        else: 
            self.set_InferenceSess_Options(model_name, backend)
            self.sess = ort.InferenceSession(model_onnx, sess_options = self.options, providers=self.providers)

    def set_iobinding (self, input): 
       
        input_name = [i.name for i in self.sess.get_inputs()]
        print (input_name)
        output_name = [i.name for i in self.sess.get_outputs()]
        self.sess_input = {}
        self.io_binding = self.sess.io_binding()
        io_binding = self.io_binding
        

        for j,i in enumerate(input_name): 
            self.io_binding.bind_cpu_input(i, np.asarray(input[j]))
        for i in output_name: 
            self.io_binding.bind_output(i)
    
    def run_profile (self, batch):
        #batch = self.preproc(batch)
        self.set_iobinding (batch)
        self.options.enable_profiling = True
        
        self.sess.run_with_iobinding(self.io_binding)
    
    def postprocess (self, out_dir): 
        #self.sess.end_profiling()
        print("RACHID")
        os.system (f'mkdir -p {out_dir}')
        os.system (f'mv {self.model_name}_* {out_dir}')
    
    def analyze_profile (self,):
        pass 

class ProfileHFONNX (ProfileONNX): 
    def __init__(self, model_onnx=None, model_config=None, model_name="model", hf=False, task=None, backend=["cuda"],preproc = None,  **kwargs):
        
        super().__init__(model_onnx, model_name, True, task, backend)
        self.set_InferenceSess_Options(model_name, backend) 
        self.provider = "CPUExecutionProvider" if self.backend == 'cpu' else "CUDAExecutionProvider" #"CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
        self.model = model_onnx.from_pretrained(model_config, session_options= self.options, provider = self.provider, use_cache=kwargs['use_cache'],  use_io_binding=kwargs['use_io_binding'])
        self.preproc = self.set_preprocessor(preproc, model_config, task)
    
    def set_model (self, task, model_config): 
        if (task == "text-generation-with-past"): 
            pass 
        elif (task == "text-classification"): 
            pass 
        elif (task == 'semantic-segmentation'): 
            pass 
        elif (task == 'object-detection'): 
            pass 
        else: 
            print(f"Task {task} Not Supported yet!!")
            exit()

    def set_preprocessor (self, preprocessor, model_config, task): 
        self.preprocessor = preprocessor.from_pretrained (model_config)
        if (task == 'text-generation-with-past' or task == 'text-classification'): 
            def tokenizer (batch): 
                self.preprocessor.eos_token = ""
                self.preprocessor.bos_token = self.preprocessor.eos_token
                self.preprocessor.pad_token = self.preprocessor.bos_token
                inputs = self.preprocessor(batch, padding = True ,return_tensors="pt")
                batch_size, seq_length = inputs['input_ids'].shape
                attention_mask = inputs ['attention_mask']
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                inputs["position_ids"] = position_ids
                return inputs 
            return tokenizer
        else: 
            def img_processor (batch): 
                print (f"Debugging Inputs: {inputs}")
                inputs = self.preprocessor(batch, return_tensors="pt")
                
                return inputs
            return img_processor

    def run_profile(self, batch):
        
        batch = self.preproc(batch) if self.task !='cv' else batch
        
        out = self.model (**batch)
        return
        #self.model.model.end_profiling()

def reset (model_onnx, model_config, model_name, hf, task , backend, preproc): 
    if (not hf):
        x = ProfileONNX (model_onnx, model_name, hf, task , backend, preproc)
    else: 
        # try:
        use_cache = False if task =='cv' else True
        use_io_binding = False if task =='cv' else True
        x = ProfileHFONNX (model_onnx, model_config, model_name, True, task, backend, preproc, use_cache = use_cache, use_io_binding= use_io_binding)
            #print (f'use_cache: {use_cache}')
            
        # except: 
        #     print ('use_cache: False')
        #     x = ProfileHFONNX (model_onnx, model_config, model_name, True, task, backend, preproc, use_cache = False, use_io_binding= False)

    return x



class Dataset: 
    def __init__(self):
        pass

class Imagenet (Dataset):
    def __init__(self, path,  transform, path_ann=None):
        self.path = path
        self.path_ann = path_ann
        self.transform = transform
        self.dataset_name = "imagenet"
    def set_up_dataloader (self, batch_size, nb_samples): 
        dataset = vision_data.ImageNet(self.path,"val",transform = self.transform)
        
        subset_idx = random.sample(range(len(dataset)), nb_samples)
        subset = Subset(dataset, subset_idx)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last = False)
        return dataloader

class Coco (Dataset):
    def __init__(self, path, path_ann, transform):
        super().__init__()
        self.path = path
        self.path_ann = path_ann
        self.transform = transform
        print (transform)
        self.dataset_name = "coco"
    def set_up_dataloader (self, batch_size, nb_samples): 
        dataset = vision_data.CocoDetection(root= self.path, annFile=self.path_ann,transform = self.transform)
        subset_idx = random.sample(range(len(dataset)), nb_samples)
        subset = Subset(dataset, subset_idx)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last = False)
        return dataloader 

class Wikitext (Dataset): 
    def __init__(self):
        super().__init__()
    

def run_profile_hf_nlp (x, nb_profiles, dataloader, input_preproc, task): 
    run = 0
    for batch in dataloader: 
        inputs = input_preproc (batch,padding = True, return_tensors ='pt')
        batch_size, seq_length = inputs['input_ids'].shape
        position_ids = torch.arange(0, seq_length + 0, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0)
        inputs ['position_ids'] = position_ids
        out = x.run_profile (inputs)
        run += 1

def load_wiki (nb_samples, batch_size): 
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
    dataset_pd = dataset.to_pandas()
    print (type(dataset_pd))
    dataset = datasets.Dataset.from_pandas(dataset_pd[dataset_pd['text']!=""])
    subset_idx = random.sample(range(len(wiki)), nb_samples)
    subset = Subset(wiki, subset_idx)
    dataloader = DataLoader(subset, batch_size, shuffle=True, num_workers=0, drop_last = False)
    return dataloader

def load_imagenet (nb_samples, batch_size, path_to_dataset): 
    imagenet = load_dataset('imagefolder',  data_dir=path_to_dataset, split="validation")
    subset_idx = random.sample(range(len(imagenet)), nb_samples)
    subset = Subset(imagenet, subset_idx)
    dataloader = DataLoader(subset, batch_size, shuffle=True, num_workers=0, drop_last = False)
    return dataloader

def load_coco (nb_samples, batch_size, path_to_dataset): 
    pass 

