
EXPORT_HF_CMD =f"python3 export_onnx.py --model_name gpt2 --hf_model --backen cuda --batch_size 1 --out_dir onnx_dir "

EXPORT_CMD =f"python3 export_onnx.py --model_name swin-t  --backen cuda --batch_size 1 --out_dir onnx_dir"

RUN_PROF_HF_CMD =f"python3 eval.py --model_name gpt2 --hf_model --dataset wikitext --backend cpu cuda --nb_smpls 4 --batch_size 1 --out_dir profiled_data "

RUN_PROF_CMD =f"python3 eval.py --model_name swin-t  --dataset wikitext --dataset_path /Users/rfk/datasets/imagenet --onnx_dir onnx_dir --backend cpu cuda --nb_smpls 4 --batch_size 1 --out_dir profiled_data "