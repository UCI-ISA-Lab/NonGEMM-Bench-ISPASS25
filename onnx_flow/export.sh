PYTHON=python
BATCH_SIZE=1 
HF_MODEL_NAME='llama gpt2-xl segformer detr'
TV_MODEL_NAME='swin-t swin-b'
HF_MODEL=--hf_model
NUM_SMPLS=26
ONNX_DIR=onnx
BACKEND=cuda
OUT_DIR=non-gemm-out-$BACKEND

mkdir logs
$PYTHON epxort_cmd.py --model_name $HF_MODEL_NAME --hf_model --batch_size $BATCH_SIZE > logs/export.log
$PYTHON epxort_cmd.py --model_name $TV_MODEL_NAME  --batch_size $BATCH_SIZE > logs/export.log
