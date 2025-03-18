# Set Path to TensorRT/samples/
#export TRT_SAMPLES=/home/rachid/tensorrt/TensorRT/samples
# Set Path to TensorRT build output 
#export TRT_BUILD_OUT=/home/rachid/tensorrt/TensorRT/build/out 
# Set Path to output directory containing the summarized ou
#SUMMARY_PATH=./non-gemm-summary
# export LD_LIBRARY_PATH=/home/rachid/tensorrt/TensorRT-10.4.0.26/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH


### Running for Model: with Input Size: ###
# Set Model specific variables
MODEL_NAME=detr
MODEL_ONNX=${ONNX_DIR}/detr_onnx/model.onnx 
IN_TENSORS=pixel_values:1x3x800x1088
BATCH_SIZE=1
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
###

# ###
# IN_TENSORS=pixel_values:2x3x800x1088
# BATCH_SIZE=2
# TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
# mkdir -p ${TRT_LOG_PATH}
# python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
# python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
# python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt 
# ###


# ###
# IN_TENSORS=pixel_values:4x3x800x1088
# BATCH_SIZE=4
# TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
# mkdir -p ${TRT_LOG_PATH}
# python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
# python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
# python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt 
# ###

# ###
# IN_TENSORS=pixel_values:8x3x800x1088
# BATCH_SIZE=8
# TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
# mkdir -p ${TRT_LOG_PATH}
# python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
# python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
# python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt 
# ###