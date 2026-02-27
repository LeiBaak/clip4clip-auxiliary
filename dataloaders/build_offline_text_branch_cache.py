import os
import csv
import json
import argparse
from collections import defaultdict
import multiprocessing as mp

from dataloaders.text_branch_utils import (
    build_text_branches,
    _offline_worker_init,
    _offline_worker_build_caption,
    _offline_worker_build_caption_batch,
    _resolve_allennlp_model_path_once,
    _resolve_offline_cuda_devices,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DATASET_CHOICES = ["msrvtt", "msvd", "lsmdc", "activity", "didemo", "all"]


def _chunked(items, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _norm(s):
    return str(s).strip()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _captions_msvd(data_path, subset):
    file_path = os.path.join(data_path, "msvd_{}.json".format(subset))
    if not os.path.exists(file_path):
        return []
    data = _load_json(file_path)
    out = []
    for item in data:
        caps = item.get("caption", [])
        if isinstance(caps, str):
            caps = [caps]
        for c in caps:
            c = _norm(c)
            if c:
                out.append(c)
    return out


def _captions_msrvtt(data_path, subset, train_csv="", val_csv="", test_csv=""):
    if subset == "train":
        train_json = _load_json(data_path)
        if not _norm(train_csv) or not os.path.exists(train_csv):
            return []
        train_video_ids = set()
        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = _norm(row.get("video_id", ""))
                if vid:
                    train_video_ids.add(vid)

        # Keep exactly the same order as MSRVTT_TrainDataLoader with
        # unfold_sentences=True:
        # iterate MSRVTT_data.json['sentences'] and keep items whose video_id
        # is in train csv list.
        out = []
        for itm in train_json.get("sentences", []):
            vid = _norm(itm.get("video_id", ""))
            cap = _norm(itm.get("caption", ""))
            if vid in train_video_ids and cap:
                out.append(cap)
        return out

    csv_path = val_csv if subset == "val" else test_csv
    if not _norm(csv_path) or not os.path.exists(csv_path):
        return []

    out = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent = _norm(row.get("sentence", ""))
            if sent:
                out.append(sent)
    return out


def _captions_lsmdc(data_path, subset):
    file_map = {
        "train": "LSMDC16_annos_training.csv",
        "val": "LSMDC16_annos_val.csv",
        "test": "LSMDC16_challenge_1000_publictect.csv",
    }
    file_path = os.path.join(data_path, file_map.get(subset, ""))
    if not os.path.exists(file_path):
        return []

    out = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            sent = _norm(parts[-1])
            if sent:
                out.append(sent)
    return out


def _captions_activity(data_path, subset):
    file_map = {
        "train": "train.json",
        "val": "val_1.json",
    }
    file_path = os.path.join(data_path, file_map.get(subset, ""))
    if not os.path.exists(file_path):
        return []

    data = _load_json(file_path)
    out = []
    for _, obj in data.items():
        sents = obj.get("sentences", [])
        merged = _norm(" ".join([_norm(x) for x in sents if _norm(x)]))
        if merged:
            out.append(merged)
    return out


def _captions_didemo(data_path, subset):
    file_map = {
        "train": "train_data.json",
        "val": "val_data.json",
        "test": "test_data.json",
    }
    file_path = os.path.join(data_path, file_map.get(subset, ""))
    if not os.path.exists(file_path):
        return []

    data = _load_json(file_path)
    by_video = defaultdict(list)
    for itm in data:
        vid = _norm(itm.get("video", ""))
        desc = _norm(itm.get("description", ""))
        if vid and desc:
            by_video[vid].append(desc)

    out = []
    for _, descs in by_video.items():
        merged = _norm(" ".join(descs))
        if merged:
            out.append(merged)
    return out


def _collect_captions(dataset, subset, args):
    if dataset == "msvd":
        return _captions_msvd(args.data_path, subset)
    if dataset == "msrvtt":
        return _captions_msrvtt(args.data_path, subset, args.msrvtt_train_csv, args.msrvtt_val_csv, args.msrvtt_test_csv)
    if dataset == "lsmdc":
        return _captions_lsmdc(args.data_path, subset)
    if dataset == "activity":
        return _captions_activity(args.data_path, subset)
    if dataset == "didemo":
        return _captions_didemo(args.data_path, subset)
    return []


def _default_subsets(dataset):
    if dataset == "activity":
        return ["train", "val"]
    return ["train", "val", "test"]


def _build_one(dataset, subset, args):
    caps = _collect_captions(dataset, subset, args)
    caps = [_norm(c) for c in caps if _norm(c)]
    if len(caps) == 0:
        print("[skip] dataset={}, subset={}, no captions found".format(dataset, subset))
        return

    uniq_caps = list(dict.fromkeys(caps))

    resolved_model_path = _resolve_allennlp_model_path_once(args.allennlp_srl_model_path)
    selected_devices = _resolve_offline_cuda_devices(args.allennlp_srl_cuda_device)
    print("[info] dataset={}, subset={}, ALLENNLP_SRL_CUDA_DEVICE={}, resolved_devices={}".format(
        dataset,
        subset,
        args.allennlp_srl_cuda_device,
        ",".join(str(x) for x in selected_devices),
    ))

    extracted_unique = []
    batch_size = max(1, int(args.srl_batch_size))
    if selected_devices == [-1] or len(selected_devices) <= 1 or len(caps) < 256:
        cap_chunks = list(_chunked(uniq_caps, batch_size))
        iterator = cap_chunks
        if tqdm is not None:
            iterator = tqdm(cap_chunks, total=len(cap_chunks), desc="[{}-{}]".format(dataset, subset), dynamic_ncols=True)
        processed = 0
        for chunk in iterator:
            rows = _offline_worker_build_caption_batch(chunk)
            extracted_unique.extend(rows)
            processed += len(chunk)
            if tqdm is None and processed % 500 == 0:
                print("[{}-{}] {}/{}".format(dataset, subset, processed, len(uniq_caps)))
    else:
        ctx = mp.get_context("spawn")
        workers = max(1, int(args.workers_per_device)) * len(selected_devices)
        chunksize = max(16, len(uniq_caps) // max(workers * 32, 1))
        cap_chunks = list(_chunked(uniq_caps, batch_size))
        print("[info] dataset={}, subset={}, mode=multi-gpu, workers={}, chunksize={}".format(
            dataset,
            subset,
            workers,
            chunksize,
        ))
        with ctx.Pool(
            processes=workers,
            initializer=_offline_worker_init,
            initargs=(resolved_model_path, tuple(int(d) for d in selected_devices)),
        ) as pool:
            stream = pool.imap(_offline_worker_build_caption_batch, cap_chunks, chunksize=chunksize)
            iterator = stream
            if tqdm is not None:
                iterator = tqdm(stream, total=len(cap_chunks), desc="[{}-{}]".format(dataset, subset), dynamic_ncols=True)

            processed = 0
            for rows in iterator:
                extracted_unique.extend(rows)
                processed += len(rows)
                if tqdm is None and processed % 500 == 0:
                    print("[{}-{}] {}/{}".format(dataset, subset, processed, len(uniq_caps)))

    extracted_by_caption = {}
    for row in extracted_unique:
        cap_key = _norm(row[0])
        extracted_by_caption[cap_key] = row

    branches = {}
    records = []
    ef = 0
    af = 0
    for idx, cap_in in enumerate(caps):
        row = extracted_by_caption.get(cap_in)
        if row is None:
            raise RuntimeError("Missing extracted row for caption at idx {} for {}-{}".format(idx, dataset, subset))
        cap_out, entity_text, action_text, entity_fb, action_fb = row
        cap_out = _norm(cap_out)
        if cap_out != cap_in:
            raise RuntimeError(
                "Caption alignment mismatch at idx {} for {}-{}: '{}' != '{}'".format(
                    idx, dataset, subset, cap_out[:120], cap_in[:120]
                )
            )

        branches[cap_in] = {
            "entity_text": entity_text,
            "action_text": action_text,
            "entity_fallback": int(entity_fb),
            "action_fallback": int(action_fb),
        }
        records.append({
            "idx": idx,
            "caption": cap_in,
            "entity_text": entity_text,
            "action_text": action_text,
            "entity_fallback": int(entity_fb),
            "action_fallback": int(action_fb),
        })

        ef += int(entity_fb)
        af += int(action_fb)

    payload = {
        "meta": {
            "dataset": dataset,
            "subset": subset,
            "ordered": True,
            "num_records": len(records),
            "num_unique_captions": len(uniq_caps),
            "allennlp_srl_cuda_device": str(args.allennlp_srl_cuda_device),
            "workers_per_device": int(args.workers_per_device),
            "resolved_devices": ",".join(str(x) for x in selected_devices),
            "source_script": "dataloaders/build_offline_text_branch_cache.py",
        },
        "records": records,
        "branches": branches,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "{}_{}_text_branches.json".format(dataset, subset))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    n = max(len(uniq_caps), 1)
    print("[done] {} {} -> {} | unique={} | entity_fb={:.4f} | action_fb={:.4f}".format(
        dataset,
        subset,
        out_path,
        len(uniq_caps),
        ef / n,
        af / n,
    ))


def main():
    parser = argparse.ArgumentParser("Build offline entity/action text branch cache for retrieval datasets")
    parser.add_argument("--dataset", type=str, default="all", choices=DATASET_CHOICES)
    parser.add_argument("--data_path", type=str, required=True, help="Dataset root path (or MSRVTT json path when dataset=msrvtt)")
    parser.add_argument("--subsets", type=str, default="", help="Comma-separated subsets; empty uses dataset defaults")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for *_text_branches.json")
    parser.add_argument("--msrvtt_train_csv", type=str, default="")
    parser.add_argument("--msrvtt_val_csv", type=str, default="")
    parser.add_argument("--msrvtt_test_csv", type=str, default="")
    parser.add_argument("--allennlp_srl_model_path", type=str, default="/data/jzw/CLIP4Clip-auxiliary/text_branches/structured-prediction-srl-bert.2020.12.15.tar.gz")
    parser.add_argument("--allennlp_srl_cuda_device", type=str, default="0", help="CUDA device index used by AllenNLP SRL predictor")
    parser.add_argument("--workers_per_device", type=int, default=2, help="Offline SRL workers per visible CUDA device")
    parser.add_argument("--srl_batch_size", type=int, default=32, help="Sentences per SRL inference batch in each worker")
    args = parser.parse_args()

    os.environ["ALLENNLP_SRL_CUDA_DEVICE"] = str(args.allennlp_srl_cuda_device)
    print("[info] ALLENNLP_SRL_CUDA_DEVICE={}".format(os.environ["ALLENNLP_SRL_CUDA_DEVICE"]))

    datasets = [args.dataset] if args.dataset != "all" else ["msrvtt", "msvd", "lsmdc", "activity", "didemo"]

    for dataset in datasets:
        if _norm(args.subsets):
            subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
        else:
            subsets = _default_subsets(dataset)

        for subset in subsets:
            _build_one(dataset, subset, args)


if __name__ == "__main__":
    main()
