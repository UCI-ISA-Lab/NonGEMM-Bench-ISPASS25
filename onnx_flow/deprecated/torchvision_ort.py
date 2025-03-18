import torch 
import torchvision 
import onnx 
import onnxruntime as ort 
import os
from PIL import Image
import requests
from torchvision.models import (vit_b_16, vit_h_14, vit_l_16, ViT_B_16_Weights, ViT_H_14_Weights, ViT_L_16_Weights, swin_b, swin_s, swin_t, Swin_B_Weights, Swin_S_Weights, Swin_T_Weights, resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights)
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
import time

#models: 'name':(model, weigths, input_names, output_names, dynamic_axes, dummy_input)
torch_models = {
    'swin-b':(swin_b,Swin_B_Weights,['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224)), 
    'swin-s':(swin_s,Swin_S_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]},(1,3,224,224)), 
    'swin-t':(swin_t, Swin_T_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224)), 
    'resnet50':(resnet50, ResNet50_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224)), 
    "mobilenet":(mobilenet_v2,MobileNet_V2_Weights, ['input'],["logits"],{"input": [0, 1, 2, 3],"logits":[0,1]}, (1,3,224,224)),
    'maskrcnn': (maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,['input'], ["boxes", "labels", "scores", "masks"], {"input": [0, 1, 2, 3], "boxes": [0, 1], "labels": [0], "scores": [0], "masks": [0, 1, 2, 3],}, (1,3,224,224) ),
    'fasterrcnn':(fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,['input'],['boxes', 'labels', 'scores'],{"input": [0, 1, 2, 3], "boxes": [0, 1], "labels": [0], "scores": [0]}, (1,3,224,224))
    }

class ModeltoONNX: 
    def __init__(self, model_name,model, weights = None, input_names = None, output_names = None, dynamic_axes = None, dummy_input_shape = None, *args):
        self.model_name = model_name
        self.model = model 
        self.weights = weights 
        self.input_names = input_names 
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.dummy_input_shape = dummy_input_shape
    def export_torchvision_to_onnx (self, out_dir = None, opset_version = 18 ): # export using torch.jit, torch dynamo based export to be supported
        
        if (out_dir == None): 
            out_dir = f"onnx/"
        path_to_model = f"{out_dir}/{self.model_name}.onnx"
        os.system(f"mkdir -p {out_dir}")
        if (self.weights != None):
            model_torch = self.model (weights = self.weights.DEFAULT).eval()
        else:  
            model_torch = self.model.eval()
        
        if (self.dummy_input_shape == None): 
            print("Please Specify a dummy input to export the model to ONNX format!!")
            exit()
        dummy_input = torch.randn(self.dummy_input_shape) 

        model_onnx = torch.onnx.export(model_torch, dummy_input, path_to_model , opset_version = opset_version, input_names = self.input_names, output_names=self.output_names, dynamic_axes=self.dynamic_axes) # to be replaced by dynamo export 
        print (f"Model {self.model_name} successfully exported to onnx!")
        return path_to_model


def convert_torchvision_to_onnx (out_dir, model_list = None, *args): 
    if (model_list == None): 
        print ("Please Specify a valid model dictionary!")
        exit()
    path_to_models = []
    for key, val in model_list.items(): 
        print (f'Exporting Model {key}:')
        print (f'val: {val}')
        print (f'key {key}')
        model2onnx = ModeltoONNX(key, val[0],val[1],val[2],val[3],val[4],val[5])
        path = model2onnx.export_torchvision_to_onnx (out_dir)
        print (f'Model {key} Exported\n')
        path_to_models.append(path)
    print ('\n Export Complete \n')
    return path_to_models

def verify_onnx_torchvision_model (model, model_onnx, input): 
    out_ref = model(input)
    options = ort.SessionOptions()
    options.enable_profiling = True
    sess = ort.InferenceSession(model_onnx, sess_options = options, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = [i.name for i in sess.get_outputs()]
    print (output_name)
    
    sess_input = {input_name: np.asarray(input)}

    io_binding = sess.io_binding()
    # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
    io_binding.bind_cpu_input(input_name, np.asarray(input))
    for i in output_name: 
        io_binding.bind_output(i)
    st = time.time()
    sess.run_with_iobinding(io_binding)
    et = time.time() - st
    print (f'TIme to run models: {et}')

    st = time.time()
    Y = io_binding.copy_outputs_to_cpu()
    et = time.time() - st
    print (f'TIme to copy outputs: {et}')
    #print (f"binded output: {Y}")
    st = time.time()
    #out_onnx = sess.run(output_name, sess_input)
    et = time.time() - st
    print (f"TIME No IO binding: {et}")

    return Y, out_ref

def draw_boxes_rcnn (ref, boxes, image):
    fig = plt.figure() 
    rows, cols = (1,4)
    
    tr = torchvision.transforms.ToTensor()
    image_np = image.convert('RGB')
    fig.add_subplot(rows, cols, 1)
    plt.imshow(image_np)
    
    image = (tr(image_np) * 255).to(torch.uint8)
    img_ref = draw_bounding_boxes(image, boxes=ref[0]['boxes'], width=4)
    img = draw_bounding_boxes(image, boxes=torch.from_numpy(boxes), width=4)
    fig.add_subplot(rows, cols, 2)
    c= plt.imshow(img_ref.permute(1, 2, 0))
    fig.add_subplot(rows, cols, 3)
    d= plt.imshow(img.permute(1, 2, 0))
    
    plt.show()
      


if __name__ == "__main__": 
    
    #save_dir = "/Users/rfk/non_gemm_op/models/HF_Transformers/playground /onnx/swin-t_onnx"
    #model_onnx = onnx.load(f'{save_dir}/swin-t.onnx')
    #model_list = {'maskrcnn': (maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,['input'], ["boxes", "labels", "scores", "masks"], {"input": [0, 1, 2, 3], "boxes": [0, 1], "labels": [0], "scores": [0], "masks": [0, 1, 2, 3],}, (1,3,224,224) ),}
    #out_dir = 'onnx/test'
    #path_to_models= convert_torchvision_to_onnx(out_dir, model_list)
    model_onnx = "/Users/rfk/non_gemm_op/models/HF_Transformers/playground /onnx/swin-t_onnx/swin-t.onnx"
    weights = Swin_T_Weights.DEFAULT
    model_ref = swin_t(weights = weights).eval()
    url ="http://images.cocodataset.org/val2017/000000397133.jpg" # "http://images.cocodataset.org/val2017/000000037777.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    transform = weights.transforms()
    image_transformed = transform(image)
    
    image_transformed2 = transform (image)
    print (image_transformed.shape)
    inputs = torch.cat([ image_transformed.unsqueeze(0),image_transformed.unsqueeze(0)])
    print (inputs.shape)
    out, out_ref = verify_onnx_torchvision_model (model_ref, model_onnx, torch.cat([ image_transformed.unsqueeze(0),image_transformed.unsqueeze(0)]))
    #boxes, labels, scores, masks = out

    print (f' Output: {out}')
    print (f'\n')
    print (f' REFERENCE Output: {out_ref}')
    #draw_boxes_rcnn(out_ref, boxes, image)
