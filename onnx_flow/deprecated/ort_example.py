import torch 
import transformers 
import torchvision 
from torchvision.models import resnet50, vit_b_16, swin_t 
import onnxruntime as ort 
from transformers import onnx
import optimum.onnxruntime as op 
from PIL import Image
import requests
import os
import datasets 
from torch.utils.data import Subset, DataLoader
import random
seed = 34

def run_onnx_gpt (model_onnx, input_seq, tokenizer):
    inputs = tokenizer(input_seq, padding = True ,return_tensors="pt")
    batch_size, seq_length = inputs['input_ids'].shape
    position_ids = torch.arange(0, seq_length + 0, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)
    inputs ['position_ids'] = position_ids
    out = model_onnx (**inputs)
    return out
def run_onnx_llama (model_onnx, input_seq, tokenizer): 
    inputs = tokenizer(input_seq, padding = True ,return_tensors="pt")
    batch_size, seq_length = inputs['input_ids'].shape
    position_ids = torch.arange(0, seq_length + 0, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)
    inputs ['position_ids'] = position_ids
    out = model_onnx (**inputs)
    return out
def run_onnx_bert (model_onnx, input_seq, tokenizer): 
    inputs = tokenizer(input_seq, padding = True ,return_tensors="pt")
    batch_size, seq_length = inputs['input_ids'].shape
    position_ids = torch.arange(0, seq_length + 0, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)
    inputs ['position_ids'] = position_ids
    out = model_onnx (**inputs)
    return out
def run_onnx_detr (model_onnx, input_images, img_processor): 
    inputs = img_processor(input_images, return_tensors="pt")
    out = model_onnx (**input_images)
    return out
def run_onnx_segformer (model_onnx, input_images, img_processor): 
    inputs = img_processor(input_images, return_tensors="pt")
    out = model_onnx (**input_images)
    return out


# "7b" : "/Users/rfk/llama/llama/llama-2-7b-hf/", # path to weights
#model_hf = transformers.AutoModelForCausalLM.from_pretrained('gpt2').eval()

save_dir = "/Users/rfk/non_gemm_op/models/HF_Transformers/playground /onnx/hf/test/gpt2_onnx"
options = ort.SessionOptions()

wiki = datasets.load_dataset("wikitext","wikitext-2-raw-v1",  split = "test")
subset_idx = random.sample(range(len(wiki)), 4)
subset = Subset(wiki, subset_idx)



#save_dir = "/Users/rfk/non_gemm_op/models/HF_Transformers/playground /onnx/llama_onnx"
tokenizer = transformers.AutoTokenizer.from_pretrained(save_dir) 
tokenizer.eos_token = "<|endoftext|>"
tokenizer.bos_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.bos_token
options = ort.SessionOptions()
#options.enable_profiling = True
#options.profile_file_prefix ="Test"

ort_model_hf = op.ORTModelForCausalLM.from_pretrained(save_dir,session_options= options, use_cache = True, use_io_binding=True,)# , export=True)
#ort_model_hf.save_pretrained (save_dir)
print(type(ort_model_hf.model))

input_seq = "Bye"
dataloader = DataLoader(subset, batch_size=4, shuffle=True, num_workers=0, drop_last = False)
for batch in dataloader: 
    #print (type(batch))
    print(type(batch))
    print (len(batch['text']))
    print (batch)
    #inp = tokenizer ([input_seq, "My mother told me"],padding = True ,return_tensors="pt")
    #print (inp)
    
    out = run_onnx_gpt (ort_model_hf, batch['text'], tokenizer)#, "My mother told me", "dalkjdhjlk fdsdf "
    break
exit()
inputs = {}
inputs['input_ids'] = torch.randn(1,5)
inputs['attention_mask'] = torch.randn(1,5)

position_ids = torch.arange(0, inputs['input_ids'].shape[-1] + 0, dtype=torch.long)
position_ids = position_ids.unsqueeze(0)
print (position_ids)


input_seq = "At the halfway point of the season , with the Blue Jackets barely into double digit wins with an 11 25  5 record , worst in the league , and sitting 20 points out of playoff position , Columbus fired Arniel . He was replaced by Assistant Coach Todd Richards on an interim basis . Richards had previously coached the Minnesota Wild . He recorded his first coaching victory for the Blue Jackets in his second game , a 4 3 win over the Phoenix Coyotes . The change in coaching did not change the fortunes of the team , as they reached the All @-@ Star break with a 13 30  6 record . At the break , Blue Jackets ' owner John P. McConnell sent out a letter to fans stating his understanding of their frustration . He added that action would be taken around the trade deadline , the Entry Draft and free agency to take the team in a new direction . When speaking of the season , McConnell stated  disappointing is not a strong enough word  and that he was committed to giving fans a team of which they can be proud of . He also thanked them for their dedication and passion , while reiterating that the team goal was to  win consistently and compete for the Stanley Cup .  Days later , a 250 @-@ person protest occurred outside of Nationwide Arena . Fans were upset with the Blue Jackets ' management and were calling for changes at the top . The same day the fans protested , it was announced that the franchise would host the 2013 All @-@ Star Game . Columbus was without a representative for the 2012 All @-@ star Game , but Ryan Johansen represented the club as a rookie participant in the Super Skills Competition . In the competition , Johansen participated in the Allstate Insurance NHL Breakaway Challenge , a shootout themed event judged by the fans . He received just 1 of the vote and finished last ."
#input_seq = "Bye"
inputs = tokenizer([input_seq, "My mother told me"],padding = True ,return_tensors="pt")
print (inputs['input_ids'].shape)
print (f' Inputs: {inputs}')
batch_size, seq_length = inputs['input_ids'].shape
#inputs['input_ids'] = torch.randn(batch_size,5)
#inputs['attention_mask'] = torch.randn(1,5)
position_ids = torch.arange(0, seq_length + 0, dtype=torch.long)
position_ids = position_ids.unsqueeze(0)
print (position_ids)
os.system(f"cd dump")
inputs ['position_ids'] = position_ids
out_ort_value = ort_model_hf (**inputs)
ort_model_hf.model.end_profiling()
model_ref = transformers.AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval()
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased") 
inputs = tokenizer([input_seq, "My mother told me"],padding = True ,return_tensors="pt")
out_ref = model_ref(**inputs)
print(f'Outputs: {out_ort_value}')
print (f'OUtput REF: {out_ref}')

output = ort_model_hf (**inputs)
print ((output))
#gen_token = ort_model_hf.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)

