export CUDA_VISIBLE_DEVICES=${DIDEMO_GPUS:-0,1,6,7}  # 并行默认4卡；独占8卡时设 DIDEMO_GPUS=0,1,2,3,4,5,6,7
NPROC_PER_NODE=${DIDEMO_NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}
DATA_PATH=/data/jzw/Didemo

for split in train val test; do
	if [ ! -f "${DATA_PATH}/didemo_${split}_text_branches.json" ]; then
		echo "[startup_didemo] missing offline cache: ${DATA_PATH}/didemo_${split}_text_branches.json"
		echo "[startup_didemo] build it first via dataloaders/build_offline_text_branch_cache.py"
		exit 1
	fi
done

python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} --master_port=29502 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --gradient_accumulation_steps=2 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compressed \
--output_dir ckpts/didemo_fulltrain \
--text_branch_cache_path ${DATA_PATH} \
--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 16 \
--datatype didemo --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32
