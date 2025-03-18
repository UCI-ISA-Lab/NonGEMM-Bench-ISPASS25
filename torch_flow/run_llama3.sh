LLAMA_WEIGHTS="None"

python run.py --model_name llama3 --model_weights ${LLAMA_WEIGHTS}  --batch_size 1 --seq_len 512  --device cuda

python run.py --model_name llama3 --model_weights ${LLAMA_WEIGHTS}  --batch_size 1 --seq_len 1024  --device cuda 

python run.py --model_name llama3 --model_weights ${LLAMA_WEIGHTS}  --batch_size 1 --seq_len 4096  --device cuda 

python run.py --model_name llama3 --model_weights ${LLAMA_WEIGHTS}  --batch_size 1 --seq_len 8192  --device cuda