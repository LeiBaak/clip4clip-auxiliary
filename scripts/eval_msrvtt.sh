export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU编号，如 0 或 1,2
DATA_PATH=/data/jzw/MSRVTT
CKPT_PATH=${1:-$(ls -1t ckpts/msrvtt_not_freeze/pytorch_model.bin.* 2>/dev/null | head -n 1)}

if [ -z "${CKPT_PATH}" ]; then
	echo "[eval_msrvtt] No checkpoint found under ckpts/msrvtt_not_freeze"
	exit 1
fi

for split in train val test; do
	if [ ! -f "${DATA_PATH}/msrvtt_${split}_text_branches.json" ]; then
		echo "[eval_msrvtt] missing offline cache: ${DATA_PATH}/msrvtt_${split}_text_branches.json"
		exit 1
	fi
done

echo "[eval_msrvtt] Use checkpoint: ${CKPT_PATH}"

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 \
main_task_retrieval.py --do_eval --num_thread_reader=8 \
--epochs=10 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/Videos_compressed \
--output_dir ckpts/msrvtt_eval \
--text_branch_cache_path ${DATA_PATH} \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 512 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--init_model ${CKPT_PATH}
