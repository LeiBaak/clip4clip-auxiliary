export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU编号，如 0 或 1,2
DATA_PATH=/data/jzw/msvd
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 \
main_task_retrieval.py --do_eval --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/compressed \
--output_dir ckpts/msvd_eval \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 512 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--init_model /data/jzw/CLIP4Clip-origin/ckpts/msvd/pytorch_model.bin.4