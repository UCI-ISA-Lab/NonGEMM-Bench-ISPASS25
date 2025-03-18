### Running for Model: with Input Size: 1 x 3 x 224 x 224 ###
# Set Model specific variables
MODEL_NAME=segformer
MODEL_ONNX=${ONNX_DIR}/${MODEL_NAME}_onnx/model.onnx #/home/rachid/ONNX-Profiling/Profiling/onnx_dir/segformer-b0_onnx/model.onnx 
IN_TENSORS=pixel_values:1x3x512x512
BATCH_SIZE=1
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 

python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
###

### Running for Model: with Input Size: 1 x 3 x 224 x 224 ###
IN_TENSORS=pixel_values:2x3x512x512
BATCH_SIZE=2
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
###

### Running for Model: with Input Size: 1 x 3 x 224 x 224 ###
IN_TENSORS=pixel_values:4x3x512x512
BATCH_SIZE=4
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
###

### Running for Model: with Input Size: 1 x 3 x 224 x 224 ###
IN_TENSORS=pixel_values:8x3x512x512
BATCH_SIZE=8
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
###
