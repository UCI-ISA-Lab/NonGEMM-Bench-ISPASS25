###'swin-base', 'swin-small', 'swin-tiny', 'vit-huge', 'vit-large', 'vit-base', 'swin-hf-large', 'swin-hf-base', 'swin-hf-small', 'vit-hf-huge', 'vit-hf-large', 'vit-hf-base', 'detr', 'maskformer-base', 'maskformer-small','segformer', 'llama2-awq', 'llama2', 'gpt2-xl', 'gpt2-large', 'gpt2', 'bert', 'maskrcnn', 'fasterrcnn',###


##VIT BASE ##
python run.py --model_name vit-base --batch_size 1 --device cuda 
python run.py --model_name vit-base --batch_size 2 --device cuda 
python run.py --model_name vit-base --batch_size 4 --device cuda 
python run.py --model_name vit-base --batch_size 8 --device cuda 

python run.py --model_name vit-base --batch_size 1 --device cpu 
python run.py --model_name vit-base --batch_size 2 --device cpu 
python run.py --model_name vit-base --batch_size 4 --device cpu 
python run.py --model_name vit-base --batch_size 8 --device cpu 


## vit-large ##
python run.py --model_name vit-large --batch_size 1 --device cuda 
python run.py --model_name vit-large --batch_size 2 --device cuda 
python run.py --model_name vit-large --batch_size 4 --device cuda 
python run.py --model_name vit-large --batch_size 8 --device cuda 

python run.py --model_name vit-large --batch_size 1 --device cpu 
python run.py --model_name vit-large --batch_size 2 --device cpu 
python run.py --model_name vit-large --batch_size 4 --device cpu 
python run.py --model_name vit-large --batch_size 8 --device cpu 

## VIT HUGE ##
python run.py --model_name vit-huge --batch_size 1 --device cuda 
python run.py --model_name vit-huge --batch_size 2 --device cuda 
python run.py --model_name vit-huge --batch_size 4 --device cuda 
python run.py --model_name vit-huge --batch_size 8 --device cuda 

python run.py --model_name vit-huge --batch_size 1 --device cpu 
python run.py --model_name vit-huge --batch_size 2 --device cpu 
python run.py --model_name vit-huge --batch_size 4 --device cpu 
python run.py --model_name vit-huge --batch_size 8 --device cpu 
