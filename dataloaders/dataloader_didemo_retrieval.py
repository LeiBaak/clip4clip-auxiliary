from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
from collections import defaultdict
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.text_branch_utils import load_text_branch_records, get_text_branches_from_records

class DiDeMo_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
                branch_cache_path="",
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.branch_records = load_text_branch_records(cache_path=branch_cache_path, cache_name="didemo_records")
        self.record_index_by_feature = {}

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")

        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "train_data.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "val_data.json")
        video_json_path_dict["test"] = os.path.join(self.data_path, "test_data.json")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as f:
            json_data = json.load(f)
        for itm in json_data:
            description = itm["description"]
            times = itm["times"]
            video = itm["video"]
            if video not in video_ids:
                continue

            # each video is split into 5-second temporal chunks
            # average the points from each annotator
            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
            if video in caption_dict:
                caption_dict[video]["start"].append(start_)
                caption_dict[video]["end"].append(end_)
                caption_dict[video]["text"].append(description)
            else:
                caption_dict[video] = {}
                caption_dict[video]["start"] = [start_]
                caption_dict[video]["end"] = [end_]
                caption_dict[video]["text"] = [description]

        for k_ in caption_dict.keys():
            caption_dict[k_]["start"] = [0]
            # trick to save time on obtaining each video length
            # [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
            # Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
            caption_dict[k_]["end"] = [31]
            caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

        cache_file = os.path.join(self.data_path, "didemo_{}_index_cache.json".format(self.subset))
        video_dict = None
        iter_pairs = None

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_obj = json.load(f)
                if isinstance(cache_obj, dict) and cache_obj.get("features_path") == self.features_path:
                    cached_video_dict = cache_obj.get("video_dict", {})
                    cached_iter_pairs = cache_obj.get("iter_pairs", [])
                    if isinstance(cached_video_dict, dict) and isinstance(cached_iter_pairs, list):
                        video_dict = cached_video_dict
                        iter_pairs = [(str(v), int(s)) for v, s in cached_iter_pairs]
            except Exception:
                video_dict = None
                iter_pairs = None

        if video_dict is None or iter_pairs is None:
            video_dict = {}
            for root, dub_dir, video_files in os.walk(self.features_path):
                for video_file in video_files:
                    file_path_ = os.path.join(root, video_file)
                    matched_ids = []
                    if video_file in video_ids:
                        matched_ids.append(video_file)
                    if video_file.endswith('.mp4'):
                        raw_name = video_file[:-4]
                        if raw_name in video_ids:
                            matched_ids.append(raw_name)
                    for video_id_ in matched_ids:
                        if video_id_ not in video_dict:
                            video_dict[video_id_] = file_path_

            valid_video_ids = list(set(video_ids) & set(caption_dict.keys()) & set(video_dict.keys()))
            valid_video_id_set = set(valid_video_ids)
            iter_pairs = []
            for video_id in caption_dict.keys():
                if video_id not in valid_video_id_set:
                    continue
                n_caption = len(caption_dict[video_id]['start'])
                for sub_id in range(n_caption):
                    iter_pairs.append((video_id, sub_id))

            try:
                cache_obj = {
                    "features_path": self.features_path,
                    "video_dict": video_dict,
                    "iter_pairs": [[v, int(s)] for v, s in iter_pairs],
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_obj, f, ensure_ascii=False)
            except Exception:
                pass

        self.caption_dict = caption_dict
        self.video_dict = video_dict
        self.iter2video_pairs_dict = {}
        for pair in iter_pairs:
            self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = pair

        self._build_record_index_mapping()

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    @staticmethod
    def _norm_text(text):
        if not isinstance(text, str):
            return ""
        return " ".join(text.strip().split())

    def _build_record_index_mapping(self):
        if not isinstance(self.branch_records, list) or len(self.branch_records) == 0:
            self.record_index_by_feature = {}
            return

        caption_to_record_indices = defaultdict(list)
        for ridx, rec in enumerate(self.branch_records):
            if not isinstance(rec, dict):
                continue
            cap = self._norm_text(rec.get("caption", ""))
            if cap:
                caption_to_record_indices[cap].append(ridx)

        caption_used_count = defaultdict(int)
        mapping = {}
        missing = 0
        for feature_idx in range(len(self.iter2video_pairs_dict)):
            video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
            sentence = self._norm_text(self.caption_dict[video_id]['text'][sub_id])
            used = caption_used_count[sentence]
            candidates = caption_to_record_indices.get(sentence, [])
            if used >= len(candidates):
                missing += 1
                continue
            mapping[feature_idx] = candidates[used]
            caption_used_count[sentence] += 1

        if missing > 0:
            raise RuntimeError(
                "didemo_records mapping failed: {} samples have no matching caption record (records={}, samples={})".format(
                    missing, len(self.branch_records), len(self.iter2video_pairs_dict)
                )
            )

        self.record_index_by_feature = mapping

    def _get_text(self, video_id, sub_id):
        caption = self.caption_dict[video_id]
        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k, dtype=np.long)
        ends = np.zeros(k, dtype=np.long)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            ind = r_ind[i]
            start_, end_ = caption['start'][ind], caption['end'][ind]
            words = self.tokenizer.tokenize(caption['text'][ind])
            starts[i], ends[i] = start_, end_

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, starts, ends

    def _get_text_from_string(self, sentence):
        pairs_text = np.zeros((1, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((1, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((1, self.max_words), dtype=np.long)

        words = self.tokenizer.tokenize(sentence)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)
        return pairs_text, pairs_mask, pairs_segment

    def _get_rawvideo(self, idx, s, e):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
        video_path = self.video_dict[idx]

        try:
            for i in range(len(s)):
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                cache_id = "{}_{}_{}".format(video_path, start_time, end_time)
                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
                raw_video_data = raw_video_data['video']

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, start_time, end_time))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e, excep))
            pass
            # raise e

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, feature_idx):
        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]

        sentence = self.caption_dict[video_id]['text'][sub_id]
        record_idx = self.record_index_by_feature.get(feature_idx, feature_idx)
        text_branches = get_text_branches_from_records(self.branch_records, record_idx, sentence, require_match=True)
        pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(video_id, sub_id)
        entity_text, entity_mask, entity_segment = self._get_text_from_string(text_branches["entity_text"])
        action_text, action_mask, action_segment = self._get_text_from_string(text_branches["action_text"])
        video, video_mask = self._get_rawvideo(video_id, starts, ends)
        entity_fallback = np.array([text_branches["entity_fallback"]], dtype=np.long)
        action_fallback = np.array([text_branches["action_fallback"]], dtype=np.long)
        return (
            pairs_text, pairs_mask, pairs_segment,
            entity_text, entity_mask, entity_segment,
            action_text, action_mask, action_segment,
            entity_fallback, action_fallback,
            video, video_mask,
        )
