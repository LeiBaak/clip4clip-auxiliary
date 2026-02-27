# CLIP4Clip-auxiliary: AI coding agent instructions

## Big picture (read this first)
- Entry point is `main_task_retrieval.py`; it owns arg parsing, DDP setup, train/eval loops, checkpointing, and logging.
- Retrieval data flow is: `dataloaders/*` -> `dataloaders/data_dataloaders.py` -> `main_task_retrieval.py` -> `modules/modeling.py` -> `metrics.py`.
- This repo currently supports 3 text/video branches during retrieval:
  - Global (original CLIP4Clip path)
  - Entity (text noun-phrase branch + prototype-based visual branch; max-sim aggregation)
  - Action (text action phrase branch + temporal-diff visual branch)
- Fixed-weight fusion is searched at eval time in `metrics.search_fusion_weights` and reported in `main_task_retrieval.eval_epoch`.

## Critical workflows (use these commands)
- Train MSVD (project-local script): `bash scripts/startup_msvd.sh`
- Eval MSVD latest ckpt (project-local script): `bash scripts/eval_msvd.sh`
- Eval explicit checkpoint: `bash scripts/eval_msvd.sh ckpts/msvd/pytorch_model.bin.<epoch>`
- Train MSRVTT: `bash scripts/startup_msrvtt.sh`
- Eval MSRVTT latest ckpt: `bash scripts/eval_msrvtt.sh`
- Train DiDeMo: `bash scripts/startup_didemo.sh`
- Eval DiDeMo latest ckpt: `bash scripts/eval_didemo.sh`
- Optional preprocessing: `python preprocess/compress_video.py --input_root <raw> --output_root <compressed>`

## Runtime assumptions and debugging gotchas
- DDP is initialized at import time in `main_task_retrieval.py` (`torch.distributed.init_process_group(...)`), so run via `python -m torch.distributed.launch ...`.
- `batch_size` and `batch_size_val` must be divisible by visible GPU count (`init_device` hard-checks this).
- Checkpoints are saved as `pytorch_model.bin.<epoch>` and optimizer as `pytorch_opt.bin.<epoch>` under `--output_dir`.
- Primary logs are written to `<output_dir>/log.txt` via `util.get_logger`.

## Project-specific data conventions
- Dataloader mapping is centralized in `dataloaders/data_dataloaders.py` via `DATALOADER_DICT`.
- Retrieval datasets can return either:
  - legacy 5-tuple: `(input_ids, input_mask, segment_ids, video, video_mask)`
  - extended 13-tuple: global/entity/action text tensors + fallback flags + video tensors.
- `main_task_retrieval.train_epoch` and `eval_epoch` handle both tuple formats; keep this backward compatibility when editing loaders.
- Text branch extraction logic lives in `dataloaders/text_branch_utils.py` (`build_text_branches`).
- Offline text-branch cache is required in retrieval runs via `--text_branch_cache_path`.
- Preferred cache format is ordered `records` (caption-aligned by dataset sample order); loaders should validate alignment.

## Model and metric integration points
- Multi-branch scoring is implemented in `modules/modeling.py`:
  - `get_multi_branch_similarity_logits`
  - entity prototype matching in `_entity_prototype_similarity_logits`
  - branch loss composition in `forward` (`loss_global + lambda_entity * loss_entity + lambda_action * loss_action`).
- Branch toggles come from args in `main_task_retrieval.py`:
  - `--disable_entity_branch`, `--disable_action_branch`
  - `--lambda_entity`, `--lambda_action`, `--fusion_grid_step`
- Eval prints branch-specific R@K, fused R@K, selected fusion weights, and fallback rates.

## Change patterns that fit this repo
- Prefer minimal, local edits: add new knobs in `get_args`, thread them into model/task_config, and log them in existing logger flow.
- If changing batch structure, update all retrieval dataloaders consistently (`msrvtt/msvd/lsmdc/activitynet/didemo`) and keep legacy fallback path.
- If changing eval behavior, preserve both single-sentence and multi-sentence branches in `eval_epoch`.
- Reuse existing utility patterns (`CrossEn`, `AllGather`, `compute_metrics`, `tensor_text_to_video_metrics`) instead of introducing parallel implementations.