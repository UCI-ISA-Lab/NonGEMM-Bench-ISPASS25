### GITHUB ISSUE: https://github.com/NVIDIA/TensorRT/issues/2796

### Running for Model: with Input Size: ###
# Set Model specific variables
MODEL_NAME=swin-b
MODEL_ONNX=${ONNX_DIR}/swin-b.onnx #/home/rachid/ONNX-Profiling/Profiling/onnx_dir/swin-b.onnx
IN_TENSORS=input:1x3x224x224
#PROF_RUNS=2000
BATCH_SIZE=1
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --num_prof_runs ${PROF_RUNS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
###

###
IN_TENSORS=input:2x3x224x224
BATCH_SIZE=2
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --num_prof_runs ${PROF_RUNS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
####



###
IN_TENSORS=input:4x3x224x224
BATCH_SIZE=4
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --num_prof_runs ${PROF_RUNS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
####



###
IN_TENSORS=input:8x3x224x224
BATCH_SIZE=8
TRT_LOG_PATH=./logs/${MODEL_NAME}_trt_${BATCH_SIZE}
mkdir -p ${TRT_LOG_PATH}
python run_trt.py --model_name ${MODEL_NAME} --model_onnx_path ${MODEL_ONNX} --input_tensors ${IN_TENSORS} --num_prof_runs ${PROF_RUNS} --out_path ${TRT_LOG_PATH} > ${TRT_LOG_PATH}/log.txt 
python ${TRT_SAMPLES}/trtexec/profiler.py ${TRT_LOG_PATH}/profile.json > ${TRT_LOG_PATH}/profile_csv.csv
python analyze_trt.py --path_to_trt_logs ${TRT_LOG_PATH} --out_dir ${SUMMARY_PATH}/${MODEL_NAME} --model_name ${MODEL_NAME} --input_shape ${BATCH_SIZE} 2>&1 | tee ${TRT_LOG_PATH}/summary.txt
####