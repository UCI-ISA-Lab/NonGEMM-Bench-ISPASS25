###'swin-base', 'swin-small', 'swin-tiny', 'vit-huge', 'vit-large', 'vit-base', 'swin-hf-large', 'swin-hf-base', 'swin-hf-small', 'vit-hf-huge', 'vit-hf-large', 'vit-hf-base', 'detr', 'maskformer-base', 'maskformer-small','segformer', 'llama2-awq', 'llama2', 'gpt2-xl', 'gpt2-large', 'gpt2', 'bert', 'maskrcnn', 'fasterrcnn',###
##GPT2 ##
python run.py --model_name bert --batch_size 1 --seq_len 128  --device cuda 
python run.py --model_name bert --batch_size 2 --seq_len 128 --device cuda 
python run.py --model_name bert --batch_size 4 --seq_len 128 --device cuda 
python run.py --model_name bert --batch_size 8 --seq_len 128 --device cuda 

python run.py --model_name bert --batch_size 1 --seq_len 128  --device cpu 
python run.py --model_name bert --batch_size 2 --seq_len 128 --device cpu 
python run.py --model_name bert --batch_size 4 --seq_len 128 --device cpu 
python run.py --model_name bert --batch_size 8 --seq_len 128 --device cpu 

