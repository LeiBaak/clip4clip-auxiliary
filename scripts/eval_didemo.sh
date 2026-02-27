export CUDA_VISIBLE_DEVICES=0  # 固定使用0号GPU
DATA_PATH=/data/jzw/Didemo
CKPT_PATH=${1:-$(ls -1t ckpts/didemo_fulltrain/pytorch_model.bin.* 2>/dev/null | head -n 1)}

if [ -z "${CKPT_PATH}" ]; then
	echo "[eval_didemo] No checkpoint found under ckpts/didemo_fulltrain"
	exit 1
fi

for split in train val test; do
	if [ ! -f "${DATA_PATH}/didemo_${split}_text_branches.json" ]; then
		echo "[eval_didemo] missing offline cache: ${DATA_PATH}/didemo_${split}_text_branches.json"
		exit 1
	fi
done

echo "[eval_didemo] Use checkpoint: ${CKPT_PATH}"

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 \
main_task_retrieval.py --do_eval --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compressed \
--output_dir ckpts/didemo_eval \
--text_branch_cache_path ${DATA_PATH} \
--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 128 \
--datatype didemo --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--init_model ${CKPT_PATH}
