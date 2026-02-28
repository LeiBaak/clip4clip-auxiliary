export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU编号，如 0 或 1,2
DATA_PATH=/data/jzw/msvd
CHECK_BRANCH_ALIGNMENT=${CHECK_BRANCH_ALIGNMENT:-1}
CKPT_PATH=${1:-$(ls -1t ckpts/msvd/pytorch_model.bin.* 2>/dev/null | head -n 1)}
DISABLE_ENTITY_QUERY_ATTENTION=${DISABLE_ENTITY_QUERY_ATTENTION:-0}

EXTRA_ARGS=""
if [ "${DISABLE_ENTITY_QUERY_ATTENTION}" = "1" ]; then
	EXTRA_ARGS="--disable_entity_query_attention"
	echo "[eval_msvd] entity query-attention: DISABLED"
else
	echo "[eval_msvd] entity query-attention: ENABLED (default)"
fi

if [ -z "${CKPT_PATH}" ]; then
	echo "[eval_msvd] No checkpoint found under ckpts/msvd"
	exit 1
fi

echo "[eval_msvd] Use checkpoint: ${CKPT_PATH}"

if [ "${CHECK_BRANCH_ALIGNMENT}" = "1" ]; then
	echo "[eval_msvd] verifying ordered branch cache alignment..."
	python -m dataloaders.verify_msvd_ordered_text_branches \
		--data_path ${DATA_PATH} \
		--cache_dir ${DATA_PATH} \
		--subsets val,test || exit 1
fi

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 \
main_task_retrieval.py --do_eval --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/compressed \
--output_dir ckpts/msvd_eval \
--text_branch_cache_path ${DATA_PATH} \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 512 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--init_model ${CKPT_PATH} \
${EXTRA_ARGS}