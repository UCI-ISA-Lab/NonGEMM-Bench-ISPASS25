bash run_segformer.sh 
echo "Segformer Profiled Successfully" 

bash run_maskformer.sh 
echo "Maskformer Profiled Successfully" 

# bash run_llama_awq.sh 
# echo "Llama2 AWQ profiled successfully" 

bash run_bert.sh 
echo "Bert Profiled Successfully" 

bash run_vit.sh 
echo "Vit Profiled Successfully" 

bash run_swin.sh 
echo "Swin Profiled Successfully" 

bash run_rcnn.sh 
echo "RCNN Profiled Successfully" 

bash run_llama.sh 
echo "Llama Profiled Successfully" 

bash run_llama3.sh 
echo "Llama3 Profiled Successfully" 

bash run_llama3_8bit.sh 
echo "Llama3-8bit Profiled Successfully" 

bash run_gpt2_xl.sh 
echo "GPT2-XL Profiled Successully" 

bash run_gpt2.sh
ehco "GPT2 profiled successfully" 

bash run_gpt2_large.sh
echo "GPT2 Large profiled successfully" 

python gen_figure_data.py



