import torch
import os
from .registry import torch_models, hf_models

class ModeltoONNX: 
    def __init__(self, model_name,model, weights = None, input_names = None, output_names = None, dynamic_axes = None, dummy_input_shape = None, *args):
        self.model_name = model_name
        self.model = model 
        self.weights = weights 
        self.input_names = input_names 
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.dummy_input_shape = dummy_input_shape
    def export_torchvision_to_onnx (self, out_dir = None, opset_version = 18 ): 
        
        if (out_dir == None): 
            out_dir = f"onnx/"
        path_to_model = f"{out_dir}/{self.model_name}.onnx"
        os.system(f"mkdir -p {out_dir}")
        if (self.weights != None):
            model_torch = self.model (weights = self.weights.DEFAULT).eval().to(torch.float16)
        else:  
            model_torch = self.model.eval()
        
        if (self.dummy_input_shape == None): 
            print("Please Specify a dummy input to export the model to ONNX format!!")
            exit()
        dummy_input = torch.randn(self.dummy_input_shape).to(torch.float16)

        model_onnx = torch.onnx.export(model_torch, dummy_input,path_to_model , opset_version = opset_version, input_names = self.input_names, output_names=self.output_names, dynamic_axes=self.dynamic_axes) # to be replaced by dynamo export 
        print (f"Model {self.model_name} successfully exported to onnx!")
        return path_to_model


    
def _convert_model_onnx (model_name, out_dir): 
    print (f'Exporting Model {model_name}:')
    val = torch_models[model_name]
    model2onnx = ModeltoONNX(model_name, val[0],val[1],val[2],val[3],val[4],val[5])
    path = model2onnx.export_torchvision_to_onnx (out_dir)
    print (f'Model {model_name} Exported\n')
    return path
def convert_models_onnx (model_list, out_dir): 
    path_to_models = {}
    for model in model_list: 
        path = _convert_model_onnx(model, out_dir)
        path_to_models [model] = path
    return path_to_models


def convert_hf_to_onnx (model_list = None, _out_dir = None, batch_size = None, dev = 'cpu' ): 
    for model in model_list:
        key = model 
        value = hf_models[key] 
        print (f'Model Being Exported: {key}')
        out_dir = f"onnx/hf/{key}_onnx" if _out_dir == None else f"{_out_dir}/{key}_onnx"
        model = value[0]
        task = value [1] 
        os.system(f"mkdir -p {out_dir}")
        try:
            os.system(f"optimum-cli export onnx --model {model} --task {task} --batch_size {batch_size} --fp16 --device {dev} {out_dir}")
            print ("Model Exported!!")
        except: 
            print (f" Failed to Export: Model {model}")
            return