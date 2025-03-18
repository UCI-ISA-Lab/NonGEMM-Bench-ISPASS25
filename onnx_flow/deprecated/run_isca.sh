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

$PYTHON prof_cmd.py --model_name $HF_MODEL_NAME $HF_MODEL --batch_size  $BATCH_SIZE --num_smpls $NUM_SMPLS --onnx_dir $ONNX_DIR --backend $BACKEND --out_dir $OUT_DIR > logs/prof.log
$PYTHON prof_cmd.py --model_name $TV_MODEL_NAME  --batch_size  $BATCH_SIZE --num_smpls $NUM_SMPLS --onnx_dir $ONNX_DIR --backend $BACKEND --out_dir $OUT_DIR > logs/prof.log


CSV_FILES_DIR=summary-onnx
$PYTHON generate_csv.py --profile_dir $OUT_DIR --out_dir $CSV_FILES_DIR --device $BACKEND > logs/post.log


# BATCH_SIZE=1 
# BACKEND=cpu
# OUT_DIR=new_inference_data_$BACKEND
# $PYTHON prof_cmd.py --model_name $MODEL_NAME $HF_MODEL --batch_size  $BATCH_SIZE --num_smpls $NUM_SMPLS --onnx_dir $ONNX_DIR --backend $BACKEND --out_dir $OUT_DIR > logs/prof.log

# $PYTHON generate_csv.py --profile_dir $OUT_DIR --out_dir $CSV_FILES_DIR --device $BACKEND > logs/post.log