MIXTRAL_WEIGHTS="None"
python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 1 --seq_len 2048  --device cuda 
python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 2 --seq_len 2048 --device cuda 
python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 4 --seq_len 2048 --device cuda 
python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 8 --seq_len 2048 --device cuda 

python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 1 --seq_len 2048  --device cpu 
python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 2 --seq_len 2048 --device cpu 
python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 4 --seq_len 2048 --device cpu 
python run.py --model_name mixtral --model_weights ${MIXTRAL_WEIGHTS}  --batch_size 8 --seq_len 2048 --device cpu 