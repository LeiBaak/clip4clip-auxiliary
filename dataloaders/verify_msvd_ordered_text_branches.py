import os
import json
import argparse


def _read_video_ids(data_path, subset):
    file_path = os.path.join(data_path, "{}_list.txt".format(subset))
    if not os.path.exists(file_path):
        raise FileNotFoundError("Missing list file: {}".format(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def _read_caption_map(data_path, subset):
    file_path = os.path.join(data_path, "msvd_{}.json".format(subset))
    if not os.path.exists(file_path):
        raise FileNotFoundError("Missing caption json: {}".format(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cap_map = {}
    for item in data:
        vid = str(item.get("video_id", "")).strip()
        caps = item.get("caption", [])
        if isinstance(caps, str):
            caps = [caps]
        cap_map[vid] = [str(c).strip() for c in caps if isinstance(c, str) and str(c).strip()]
    return cap_map


def _build_ordered_pairs(data_path, subset):
    video_ids = _read_video_ids(data_path, subset)
    cap_map = _read_caption_map(data_path, subset)
    out = []
    for vid in video_ids:
        if vid not in cap_map:
            raise RuntimeError("video_id {} missing in msvd_{}.json".format(vid, subset))
        for cap in cap_map[vid]:
            out.append((vid, cap))
    return out


def _validate_subset(data_path, cache_dir, subset):
    cache_file = os.path.join(cache_dir, "msvd_{}_text_branches.json".format(subset))
    if not os.path.exists(cache_file):
        raise FileNotFoundError("Missing cache file: {}".format(cache_file))

    with open(cache_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Invalid cache format (records list required): {}".format(cache_file))

    ordered_pairs = _build_ordered_pairs(data_path, subset)
    if len(records) != len(ordered_pairs):
        raise RuntimeError(
            "Size mismatch on {}: records={} vs pairs={}".format(subset, len(records), len(ordered_pairs))
        )

    for idx, ((vid, cap), rec) in enumerate(zip(ordered_pairs, records)):
        if not isinstance(rec, dict):
            raise RuntimeError("Record type invalid at {}:{}".format(subset, idx))
        if int(rec.get("idx", -1)) != idx:
            raise RuntimeError("Index mismatch at {}:{}".format(subset, idx))
        if str(rec.get("video_id", "")).strip() != vid:
            raise RuntimeError("video_id mismatch at {}:{}".format(subset, idx))
        if str(rec.get("caption", "")).strip() != cap:
            raise RuntimeError("caption mismatch at {}:{}".format(subset, idx))
        if not str(rec.get("entity_text", "")).strip():
            raise RuntimeError("entity_text empty at {}:{}".format(subset, idx))
        if not str(rec.get("action_text", "")).strip():
            raise RuntimeError("action_text empty at {}:{}".format(subset, idx))

    print("[ok] {} aligned: {} records".format(subset, len(records)))


def main():
    parser = argparse.ArgumentParser("Verify ordered MSVD text branches alignment")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--subsets", type=str, default="train,val,test")
    args = parser.parse_args()

    subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
    for subset in subsets:
        _validate_subset(args.data_path, args.cache_dir, subset)

    print("[ok] all subsets passed")


if __name__ == "__main__":
    main()
