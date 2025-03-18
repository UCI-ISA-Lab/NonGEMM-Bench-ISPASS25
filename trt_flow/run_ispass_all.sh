source setup.sh 
SECONDS=0
bash run_swin.sh

bash run_swin_t.sh 

bash run_detr.sh 

bash run_segformer.sh 

python plot_fig7.py --prof_dir non-gemm-summary-trt
echo "Execution time: $SECONDS seconds"