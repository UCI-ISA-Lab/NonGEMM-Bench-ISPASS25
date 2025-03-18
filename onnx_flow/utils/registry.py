import torch
import os
from torchvision.models import (vit_b_16, vit_h_14, vit_l_16, ViT_B_16_Weights, ViT_H_14_Weights, ViT_L_16_Weights, swin_b, swin_s, swin_t, Swin_B_Weights, Swin_S_Weights, Swin_T_Weights, resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights)
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights
from transformers import (AutoTokenizer, AutoImageProcessor)
from optimum.onnxruntime import (ORTModelForCausalLM, ORTModelForTokenClassification, ORTModelForSemanticSegmentation, ORTModelForCustomTasks, ORTModelForImageClassification)

class preprocessor: 
    def __init__(self):
        pass
        
    def __call__ (input: torch.tensor): 
        pass
class CVpreprocessor (preprocessor):
    def __init__(self, weight_transform):
        super().__init__()
        self.transform = weight_transform
    def __call__(self, input) -> torch.tensor:
        print (type(input))
        if (type(input) == list ): 
            batch = []
            for j,i in enumerate(input): 
                x = self.transform(i.resize((640,480)))
                x = i.unsqueeze(0)
                batch = torch.cat (batch, x) if (j > 0) else x
            return [batch]
        else:
            x = self.transform(input.resize((640,480)))
            x = x.unsqueeze(0)
            return x
  
#models: 'name':(model, weigths, input_names, output_names, dynamic_axes, input preprocessor function)
torch_models = {
    'swin-b':(swin_b,Swin_B_Weights,['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224), CVpreprocessor(Swin_B_Weights.DEFAULT.transforms())), 
    'swin-s':(swin_s,Swin_S_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]},(1,3,224,224), CVpreprocessor(Swin_S_Weights.DEFAULT.transforms())), 
    'swin-t':(swin_t, Swin_T_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224), CVpreprocessor(Swin_T_Weights.DEFAULT.transforms())), 
    'resnet50':(resnet50, ResNet50_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224), CVpreprocessor(ResNet50_Weights.DEFAULT.transforms())), 
    "mobilenet":(mobilenet_v2,MobileNet_V2_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224), CVpreprocessor(MobileNet_V2_Weights.DEFAULT.transforms())),
    'maskrcnn': (maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,['input'], ["boxes", "labels", "scores", "masks"], {"input": [0, 1, 2, 3], "boxes": [0, 1], "labels": [0], "scores": [0], "masks": [0, 1, 2, 3],}, (1,3,224,224), CVpreprocessor(MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms())),
    'fasterrcnn':(fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,['input'],['boxes', 'labels', 'scores'],{"input": [0, 1, 2, 3], "boxes": [0, 1], "labels": [0], "scores": [0]}, (1,3,224,224), CVpreprocessor(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()))
    }

#hf_models : 'name':(model, model_config, task, input preprocessor function, optimum.onnxruntime Model Class)

path_llama = "/path/to/llama-weights/llama/llama-2-7b-hf/" 
path_llama = os.getenv("LLAMA_WEIGHTS")
# path_llama = "meta-llama/Llama-2-7b-chat-hf"
hf_models = {"gpt2":("gpt2",'text-generation-with-past', AutoTokenizer, ORTModelForCausalLM), 
             "gpt2-large":("gpt2-large",'text-generation-with-past', AutoTokenizer, ORTModelForCausalLM), 
             "gpt2-xl":("gpt2-xl",'text-generation-with-past', AutoTokenizer, ORTModelForCausalLM), 
             'llama':(path_llama, 'text-generation-with-past', AutoTokenizer, ORTModelForCausalLM), 
             'segformer':('nvidia/segformer-b0-finetuned-ade-512-512', 'semantic-segmentation', AutoImageProcessor, ORTModelForSemanticSegmentation),
             'segformer-b5': ('nvidia/segformer-b5-finetuned-ade-512-512', 'semantic-segmentation', AutoImageProcessor, ORTModelForSemanticSegmentation),
             'detr':('facebook/detr-resnet-50', 'object-detection', AutoImageProcessor, ORTModelForCustomTasks),
             'bert':('bert-base-uncased', 'text-classification', AutoTokenizer, ORTModelForTokenClassification),
             'maskformer':('facebook/maskformer-swin-base-coco','semantic-segmentation',AutoImageProcessor, ORTModelForSemanticSegmentation)
             }

# ONNX models should be stored using the following format: key_onnx/key.onnx e.g. swin-t_onnx/swin-t.onnx, gpt2-large_onnx/gpt2-large.onnx
