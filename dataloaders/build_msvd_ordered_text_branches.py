import os
import json
import argparse
import multiprocessing as mp

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from dataloaders.text_branch_utils import (
    _offline_worker_init,
    _offline_worker_build_caption,
    _resolve_allennlp_model_path_once,
    _resolve_offline_cuda_devices,
)


def _load_video_ids(data_path, subset):
    list_path = os.path.join(data_path, "{}_list.txt".format(subset))
    with open(list_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def _load_caption_map(data_path, subset):
    json_file = os.path.join(data_path, "msvd_{}.json".format(subset))
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    caption_map = {}
    for item in data:
        vid = str(item.get("video_id", "")).strip()
        caps = item.get("caption", [])
        if isinstance(caps, str):
            caps = [caps]
        cap_list = [str(c).strip() for c in caps if isinstance(c, str) and str(c).strip()]
        caption_map[vid] = cap_list
    return caption_map


def _build_ordered_pairs(data_path, subset):
    video_ids = _load_video_ids(data_path, subset)
    caption_map = _load_caption_map(data_path, subset)

    pairs = []
    for vid in video_ids:
        if vid not in caption_map:
            raise RuntimeError("video_id {} not found in msvd_{}.json".format(vid, subset))
        for cap in caption_map[vid]:
            pairs.append((vid, cap))
    return pairs


def _extract_in_order(captions, model_path, devices):
    if len(captions) == 0:
        return []

    workers = len(devices)
    chunksize = max(16, len(captions) // max(workers * 32, 1))
    ctx = mp.get_context("spawn")

    with ctx.Pool(
        processes=workers,
        initializer=_offline_worker_init,
        initargs=(model_path, tuple(int(d) for d in devices)),
    ) as pool:
        stream = pool.imap(_offline_worker_build_caption, captions, chunksize=chunksize)

        out = []
        iterator = stream
        if tqdm is not None:
            iterator = tqdm(stream, total=len(captions), desc="[msvd-ordered]", dynamic_ncols=True)

        for result in iterator:
            out.append(result)

    return out


def main():
    parser = argparse.ArgumentParser("Build ordered MSVD entity/action text branches (1:1 with dataset caption order)")
    parser.add_argument("--data_path", type=str, required=True, help="MSVD root path containing train/val/test list and json files")
    parser.add_argument("--subsets", type=str, default="train,val,test", help="Comma separated subsets")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--output_suffix", type=str, default="_text_branches.json")
    parser.add_argument("--allennlp_srl_model_path", type=str, default="/data/jzw/CLIP4Clip-auxiliary/text_branches/structured-prediction-srl-bert.2020.12.15.tar.gz")
    parser.add_argument("--allennlp_srl_cuda_device", type=str, default="0,1,6,7")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_path = _resolve_allennlp_model_path_once(args.allennlp_srl_model_path)
    devices = _resolve_offline_cuda_devices(args.allennlp_srl_cuda_device)
    print("[info] offline_srl_devices={}".format(",".join(str(x) for x in devices)))
    print("[info] allennlp_srl_model_path={}".format(model_path))

    subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
    for subset in subsets:
        ordered_pairs = _build_ordered_pairs(args.data_path, subset)
        captions = [cap for _, cap in ordered_pairs]
        print("[info] subset={}, ordered_caption_count={}".format(subset, len(captions)))

        extracted = _extract_in_order(captions, model_path, devices)
        if len(extracted) != len(ordered_pairs):
            raise RuntimeError("Extraction size mismatch for subset {}".format(subset))

        records = []
        ef = 0
        af = 0
        for idx, ((vid, cap), row) in enumerate(zip(ordered_pairs, extracted)):
            cap_ret, entity_text, action_text, entity_fb, action_fb = row
            if str(cap_ret).strip() != str(cap).strip():
                raise RuntimeError("Caption order mismatch at idx {} in subset {}".format(idx, subset))

            rec = {
                "idx": idx,
                "video_id": vid,
                "caption": cap,
                "entity_text": entity_text,
                "action_text": action_text,
                "entity_fallback": int(entity_fb),
                "action_fallback": int(action_fb),
            }
            ef += int(entity_fb)
            af += int(action_fb)
            records.append(rec)

        payload = {
            "meta": {
                "dataset": "msvd",
                "subset": subset,
                "ordered": True,
                "num_records": len(records),
                "srl_backend": "allennlp",
                "allennlp_srl_model_path": args.allennlp_srl_model_path,
                "allennlp_srl_cuda_device": ",".join(str(x) for x in devices),
                "source_script": "dataloaders/build_msvd_ordered_text_branches.py",
            },
            "records": records,
        }

        out_file = os.path.join(args.output_dir, "msvd_{}{}".format(subset, args.output_suffix))
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        n = max(len(records), 1)
        print("[done] {} -> {} | records={} | entity_fb={:.4f} | action_fb={:.4f}".format(
            subset,
            out_file,
            len(records),
            ef / n,
            af / n,
        ))


if __name__ == "__main__":
    main()
