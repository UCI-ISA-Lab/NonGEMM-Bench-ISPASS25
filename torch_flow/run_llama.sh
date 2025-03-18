python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 1 --seq_len 2048  --device cuda 
python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 2 --seq_len 2048 --device cuda 
python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 4 --seq_len 2048 --device cuda 
python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 8 --seq_len 2048 --device cuda 

python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 1 --seq_len 2048  --device cpu 
python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 2 --seq_len 2048 --device cpu 
python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 4 --seq_len 2048 --device cpu 
python run.py --model_name llama2 --model_weights ${LLAMA_WEIGHTS}  --batch_size 8 --seq_len 2048 --device cpu 

