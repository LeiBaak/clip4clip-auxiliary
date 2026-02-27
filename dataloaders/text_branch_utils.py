import re
import os
import json
import argparse
import multiprocessing as mp

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "in", "on", "at", "to", "for", "from", "with", "of", "by", "as",
    "and", "or", "but", "if", "then", "while", "so", "than", "because",
    "he", "she", "it", "they", "we", "you", "i", "him", "her", "them",
    "his", "hers", "their", "its", "our", "your", "my",
    "how", "few", "lot", "same", "either",
    "someone", "something", "anything", "nothing",
    "here", "there", "then", "now", "together", "when",
    "which", "what", "who", "whom", "whose", "where", "why",
    "very", "more", "most", "less", "least",
}

_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "my", "your", "his", "her", "its", "our", "their"
}

_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "from", "with", "of", "by", "as", "into", "onto",
    "off", "over", "under", "behind", "between", "through", "across", "around", "along", "before",
    "after", "during", "without", "within", "toward", "towards", "against", "beside", "down", "up", "out", "about",
}

_AUXILIARY_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must",
}

_OBJ_PRONOUNS = {"it", "them", "him", "her", "me", "us", "someone", "something", "anything", "nothing"}
_REFLEXIVE_PRONOUNS = {"myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves"}
_SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
_INDEF_PRONOUNS = {"someone", "somebody", "something", "anyone", "anybody", "anything", "everyone", "everybody", "everything", "nobody", "nothing"}

_BOUNDARY = {"and", "or", "but", "while", "then"}
_TRAILING_FILLERS = {"here", "there", "then", "now", "together", "also", "just"}
_NOISE_ENTITY_TOKENS = {"year", "old", "like"}
_OPTIONAL_TAIL_PREP = {"for", "by", "about", "from"}

_DEFAULT_ALLENNLP_SRL_MODEL = "/data/jzw/CLIP4Clip-auxiliary/text_branches/structured-prediction-srl-bert.2020.12.15.tar.gz"
_ALLENNLP_SRL_PREDICTOR = None
_ALLENNLP_SRL_PREDICTOR_RAW = None
_ALLENNLP_SRL_INIT_FAILED = False


def _normalize_space(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _split_tokens_with_spans(text):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z']+", str(text))]


def _tokenize(text):
    toks = [tok.lower() for tok, _, _ in _split_tokens_with_spans(text)]
    return [t for t in toks if t != "s"]


def _is_candidate_noun_token(tok):
    if not tok or len(tok) <= 1:
        return False
    if tok in _STOPWORDS or tok in _PREPOSITIONS or tok in _AUXILIARY_VERBS:
        return False
    # if tok in _OBJ_PRONOUNS or tok in _REFLEXIVE_PRONOUNS or tok in _SUBJECT_PRONOUNS or tok in _INDEF_PRONOUNS:
    #     return False
    if tok in _NOISE_ENTITY_TOKENS:
        return False
    if tok.endswith("ly"):
        return False
    if tok.endswith(("ing", "ed")) and len(tok) > 4:
        return False
    return True


def _is_predicate_like(tok, prev_tok="", next_tok=""):
    if not tok or len(tok) <= 1:
        return False
    if tok in _STOPWORDS or tok in _DETERMINERS or tok in _PREPOSITIONS:
        return False
    if tok in _AUXILIARY_VERBS and tok not in {"have", "has", "had", "do", "does", "did"}:
        return False
    if tok in _OBJ_PRONOUNS or tok in _REFLEXIVE_PRONOUNS:
        return False
    if tok in {"like", "old", "young"}:
        return False
    if tok.endswith("ly"):
        return False

    if tok.endswith(("ing", "ed")):
        if tok.endswith("ing") and prev_tok in _DETERMINERS and _is_candidate_noun_token(next_tok):
            return False
        return True

    if prev_tok in _AUXILIARY_VERBS:
        return True

    if prev_tok in _SUBJECT_PRONOUNS and (next_tok in _PREPOSITIONS or _is_candidate_noun_token(next_tok) or next_tok == ""):
        return True

    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        if next_tok in _PREPOSITIONS or next_tok in _DETERMINERS or _is_candidate_noun_token(next_tok) or next_tok == "":
            return True

    if _is_candidate_noun_token(prev_tok) and (_is_candidate_noun_token(next_tok) or next_tok in _PREPOSITIONS or next_tok in _DETERMINERS):
        if tok.endswith(("er", "or", "ance", "ence", "tion", "ment", "ness", "ity", "ism")):
            return False
        return True

    return False


def _clean_phrase_tokens(tokens):
    toks = [t for t in tokens if t]
    while toks and toks[0] in _DETERMINERS:
        toks = toks[1:]
    while toks and (toks[-1] in _DETERMINERS or toks[-1] in _PREPOSITIONS or toks[-1] in _TRAILING_FILLERS):
        toks = toks[:-1]
    return toks


def _normalize_argument_tokens(tokens):
    toks = [t for t in tokens if t]
    lead_drop = _DETERMINERS | _PREPOSITIONS | _AUXILIARY_VERBS | {"and", "or", "to", "like"}
    while toks and toks[0] in lead_drop:
        toks = toks[1:]
    while toks and (toks[-1] in _DETERMINERS or toks[-1] in _PREPOSITIONS or toks[-1] in _TRAILING_FILLERS):
        toks = toks[:-1]
    return toks


def _dedup_phrases(phrases):
    seen = set()
    out = []
    for p in phrases:
        norm = _normalize_space(p)
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _join_phrases_with_and(phrases):
    merged = [p for p in phrases if _normalize_space(p)]
    if not merged:
        return ""
    if len(merged) == 1:
        return _normalize_space(merged[0])
    return " and ".join(_normalize_space(p) for p in merged)


def _trim_optional_tail_pp(phrase):
    toks = _tokenize(phrase)
    if len(toks) <= 4:
        return _normalize_space(phrase)
    cut = len(toks)
    for i in range(2, len(toks)):
        if toks[i] in _OPTIONAL_TAIL_PREP:
            cut = i
            break
    return _normalize_space(" ".join(toks[:cut]))


def _is_subsequence(short_toks, long_toks):
    if not short_toks or len(short_toks) > len(long_toks):
        return False
    i = 0
    for t in long_toks:
        if i < len(short_toks) and short_toks[i] == t:
            i += 1
    return i == len(short_toks)


def _extract_entity_phrases_rule(caption):
    tokens = _tokenize(caption)
    if not tokens:
        return []

    entities = []
    current = []

    for idx, tok in enumerate(tokens):
        prev_tok = tokens[idx - 1] if idx > 0 else ""
        next_tok = tokens[idx + 1] if idx + 1 < len(tokens) else ""

        if tok in _BOUNDARY or tok in _PREPOSITIONS or _is_predicate_like(tok, prev_tok, next_tok):
            if current:
                cleaned = _clean_phrase_tokens(current)
                if cleaned and any(_is_candidate_noun_token(t) for t in cleaned):
                    entities.append(" ".join(cleaned[-3:]))
                current = []
            continue

        if tok in _DETERMINERS:
            if current:
                cleaned = _clean_phrase_tokens(current)
                if cleaned and any(_is_candidate_noun_token(t) for t in cleaned):
                    entities.append(" ".join(cleaned[-3:]))
                current = []
            continue

        if _is_candidate_noun_token(tok):
            current.append(tok)
        elif current:
            cleaned = _clean_phrase_tokens(current)
            if cleaned and any(_is_candidate_noun_token(t) for t in cleaned):
                entities.append(" ".join(cleaned[-3:]))
            current = []

    if current:
        cleaned = _clean_phrase_tokens(current)
        if cleaned and any(_is_candidate_noun_token(t) for t in cleaned):
            entities.append(" ".join(cleaned[-3:]))

    return _dedup_phrases(entities)


def _normalize_role(role):
    role = str(role or "").upper().strip()
    if role.startswith("B-") or role.startswith("I-"):
        role = role[2:]
    return role


def _get_allennlp_srl_predictor():
    global _ALLENNLP_SRL_PREDICTOR, _ALLENNLP_SRL_PREDICTOR_RAW, _ALLENNLP_SRL_INIT_FAILED
    if _ALLENNLP_SRL_PREDICTOR is not None:
        return _ALLENNLP_SRL_PREDICTOR
    if _ALLENNLP_SRL_INIT_FAILED:
        return None

    model_path = os.environ.get("ALLENNLP_SRL_MODEL_PATH", _DEFAULT_ALLENNLP_SRL_MODEL)
    device_str = os.environ.get("ALLENNLP_SRL_CUDA_DEVICE", "0")
    try:
        cuda_device = int(device_str)
    except Exception:
        cuda_device = 0

    try:
        from allennlp.predictors.predictor import Predictor
        import allennlp_models.tagging  # noqa: F401

        predictor = Predictor.from_path(model_path, cuda_device=cuda_device)
        _ALLENNLP_SRL_PREDICTOR_RAW = predictor
        print(f"[srl-init] success cuda_device={cuda_device} model_path={model_path}")

        def _predict(sentence):
            out = predictor.predict(sentence=sentence)
            if not isinstance(out, dict):
                return {"words": [], "verbs": []}
            return {
                "words": out.get("words", []) or [],
                "verbs": out.get("verbs", []) or [],
            }

        _ALLENNLP_SRL_PREDICTOR = _predict
        return _ALLENNLP_SRL_PREDICTOR
    except Exception as e:
        print(f"[srl-init] failed cuda_device={cuda_device} model_path={model_path} error={e}")
        _ALLENNLP_SRL_INIT_FAILED = True
        return None


def _predict_srl_batch(captions):
    predictor = _get_allennlp_srl_predictor()
    if predictor is None:
        return [None for _ in captions]

    if not isinstance(captions, list):
        captions = list(captions)
    if len(captions) == 0:
        return []

    raw_predictor = _ALLENNLP_SRL_PREDICTOR_RAW
    if raw_predictor is not None:
        try:
            outputs = raw_predictor.predict_batch_json([{"sentence": c} for c in captions])
            if isinstance(outputs, list) and len(outputs) == len(captions):
                normalized = []
                for out in outputs:
                    if not isinstance(out, dict):
                        normalized.append({"words": [], "verbs": []})
                    else:
                        normalized.append({
                            "words": out.get("words", []) or [],
                            "verbs": out.get("verbs", []) or [],
                        })
                return normalized
        except Exception:
            pass

    return [predictor(c) for c in captions]


def _collect_srl_frames(srl_output):
    frames = []

    if isinstance(srl_output, dict) and isinstance(srl_output.get("words"), list) and isinstance(srl_output.get("verbs"), list):
        words = [str(w).lower() for w in srl_output.get("words", [])]
        for vb in srl_output.get("verbs", []):
            tags = vb.get("tags", []) if isinstance(vb, dict) else []
            if not isinstance(tags, list) or len(tags) != len(words):
                continue

            spans = {}
            cur_role = None
            cur = []
            for w, tag in zip(words, tags):
                role = _normalize_role(tag)
                if role == "O" or not role:
                    if cur_role and cur:
                        spans[cur_role] = " ".join(cur)
                    cur_role, cur = None, []
                    continue
                if str(tag).startswith("B-"):
                    if cur_role and cur:
                        spans[cur_role] = " ".join(cur)
                    cur_role, cur = role, [w]
                elif cur_role == role:
                    cur.append(w)
                else:
                    if cur_role and cur:
                        spans[cur_role] = " ".join(cur)
                    cur_role, cur = role, [w]
            if cur_role and cur:
                spans[cur_role] = " ".join(cur)
            if spans:
                frames.append(spans)

    elif isinstance(srl_output, list) and srl_output and isinstance(srl_output[0], dict) and "entity_group" in srl_output[0]:
        frame = {}
        for item in srl_output:
            role = _normalize_role(item.get("entity_group", ""))
            text = _normalize_space(item.get("word", "")).lower()
            if not role or not text:
                continue
            if role not in frame:
                frame[role] = text
            else:
                frame[role] = _normalize_space(frame[role] + " " + text)
        if frame:
            frames.append(frame)

    elif isinstance(srl_output, list):
        for it in srl_output:
            if not isinstance(it, dict):
                continue
            frame = {}
            verb = _normalize_space(it.get("verb", "")).lower()
            if verb:
                frame["V"] = verb
            args = it.get("arguments", [])
            if isinstance(args, list):
                for a in args:
                    if not isinstance(a, dict):
                        continue
                    role = _normalize_role(a.get("role", ""))
                    txt = _normalize_space(a.get("text", "")).lower()
                    if role and txt and role not in frame:
                        frame[role] = txt
            if frame:
                frames.append(frame)

    return frames


def _compose_semantic_phrases_from_frames(frames):
    _COPULAR_VERBS = {"is", "are", "was", "were", "be", "been", "being", "am"}

    all_frame_verbs = set()
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        vv = _normalize_space(fr.get("V", "") or fr.get("VERB", "")).lower()
        if not vv:
            continue
        for t in _tokenize(vv):
            if t:
                all_frame_verbs.add(t)

    def _object_core_tokens(tokens):
        if not tokens:
            return []
        toks = [t for t in tokens if t]
        toks = [t for t in toks if t not in all_frame_verbs]
        while toks:
            nxt = toks[1] if len(toks) > 1 else ""
            if toks[0] in _DETERMINERS or toks[0] in _PREPOSITIONS or toks[0] in _AUXILIARY_VERBS or _is_predicate_like(toks[0], "", nxt):
                toks = toks[1:]
                continue
            break
        if not toks:
            return []
        noun_like = [t for t in toks if _is_candidate_noun_token(t)]
        if noun_like:
            return noun_like[-2:]
        return toks[-2:]

    def _subject_core_tokens(tokens):
        if not tokens:
            return []
        toks = [t for t in tokens if t]

        if "and" in toks:
            idx = toks.index("and")
            left = [t for t in toks[:idx] if _is_candidate_noun_token(t)]
            right = [t for t in toks[idx + 1 :] if _is_candidate_noun_token(t)]
            if left and right:
                return [left[-1], "and", right[0]]

        first_chunk = []
        for t in toks:
            if t in _BOUNDARY or t in _PREPOSITIONS:
                break
            if _is_candidate_noun_token(t):
                first_chunk.append(t)

        if len(first_chunk) >= 2 and len(toks) <= 3:
            return first_chunk[-2:]
        if first_chunk:
            return [first_chunk[0]]

        noun_like = [t for t in toks if _is_candidate_noun_token(t)]
        if noun_like:
            return [noun_like[0]]
        return toks[:1]

    phrases = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue

        verb = _normalize_space(frame.get("V", "") or frame.get("VERB", "")).lower()
        if not verb:
            continue
        if verb in _COPULAR_VERBS:
            continue

        arg0 = _normalize_space(frame.get("ARG0", "")).lower()
        arg1 = _normalize_space(frame.get("ARG1", "")).lower()
        arg2 = _normalize_space(frame.get("ARG2", "")).lower()

        arg0_tokens = _clean_phrase_tokens(_normalize_argument_tokens(_tokenize(arg0))) if arg0 else []
        arg1_tokens = _clean_phrase_tokens(_normalize_argument_tokens(_tokenize(arg1))) if arg1 else []
        arg2_tokens = _clean_phrase_tokens(_normalize_argument_tokens(_tokenize(arg2))) if arg2 else []
        verb_tokens = _clean_phrase_tokens(_normalize_argument_tokens(_tokenize(verb)))
        arg1_object_tokens = _object_core_tokens(arg1_tokens)
        arg2_object_tokens = _object_core_tokens(arg2_tokens)
        arg0_subject_tokens = _subject_core_tokens(arg0_tokens)

        phrase_tokens = []
        if not arg0_subject_tokens and arg1_object_tokens:
            phrase_tokens.extend(arg1_object_tokens)
            phrase_tokens.extend(verb_tokens)
        elif arg1_object_tokens:
            if arg0_subject_tokens:
                phrase_tokens.extend(arg0_subject_tokens)
            phrase_tokens.extend(verb_tokens)
            phrase_tokens.extend(arg1_object_tokens)
        elif arg2_object_tokens:
            if arg0_subject_tokens:
                phrase_tokens.extend(arg0_subject_tokens)
            phrase_tokens.extend(verb_tokens)
            phrase_tokens.extend(arg2_object_tokens)
        else:
            if arg0_tokens:
                phrase_tokens.extend(arg0_subject_tokens)
            phrase_tokens.extend(verb_tokens)

        phrase_tokens = [t for t in phrase_tokens if t not in _DETERMINERS]
        phrase_tokens = _normalize_argument_tokens(phrase_tokens)
        phrase_tokens = _clean_phrase_tokens(phrase_tokens)
        if len(phrase_tokens) >= 2:
            phrases.append(" ".join(phrase_tokens))

    return _dedup_phrases(phrases)


def _entity_chunks_from_tokens(tokens):
    chunks = []
    cur = []
    for t in tokens:
        if t in _BOUNDARY or t in _PREPOSITIONS:
            if cur:
                chunks.append(cur)
            cur = []
            continue
        if _is_candidate_noun_token(t):
            cur.append(t)
        elif cur:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return [" ".join(ch[-3:]) for ch in chunks if ch]


def _parse_cuda_devices(cuda_device_arg):
    text = _normalize_space(cuda_device_arg)
    if not text:
        return [0, 1, 6, 7]
    if text == "-1":
        return [-1]

    out = []
    for part in text.split(","):
        p = _normalize_space(part)
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    if not out:
        return [0, 1, 6, 7]

    uniq = []
    seen = set()
    for d in out:
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq


def _parse_cuda_visible_devices_env():
    env = _normalize_space(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not env:
        return []

    ids = []
    for part in env.split(","):
        p = _normalize_space(part)
        if not p:
            continue
        try:
            ids.append(int(p))
        except Exception:
            continue
    return ids


def _detect_visible_cuda_devices():
    env = _normalize_space(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if env:
        ids = []
        for part in env.split(","):
            p = _normalize_space(part)
            if not p:
                continue
            try:
                ids.append(int(p))
            except Exception:
                pass
        if ids:
            # When CUDA_VISIBLE_DEVICES is set, frameworks use logical indices [0..N-1].
            return list(range(len(ids)))

    try:
        import torch
        if torch.cuda.is_available():
            return list(range(int(torch.cuda.device_count())))
    except Exception:
        pass
    return []


def _resolve_offline_cuda_devices(cuda_device_arg):
    requested = _parse_cuda_devices(cuda_device_arg)
    if requested == [-1]:
        return [-1]

    visible_env = _parse_cuda_visible_devices_env()
    if visible_env:
        phys2local = {pid: idx for idx, pid in enumerate(visible_env)}
        local_from_phys = [phys2local[d] for d in requested if d in phys2local]
        if local_from_phys:
            return local_from_phys

        local_count = len(visible_env)
        local_direct = [d for d in requested if isinstance(d, int) and 0 <= d < local_count]
        if local_direct:
            uniq = []
            seen = set()
            for d in local_direct:
                if d in seen:
                    continue
                seen.add(d)
                uniq.append(d)
            return uniq

        return list(range(local_count))

    visible = _detect_visible_cuda_devices()
    if not visible:
        return [-1]

    selected = [d for d in requested if d in visible]
    if selected:
        return selected
    return [visible[0]]


def _offline_worker_init(model_path, device_queue):
    global _ALLENNLP_SRL_PREDICTOR, _ALLENNLP_SRL_PREDICTOR_RAW, _ALLENNLP_SRL_INIT_FAILED

    device = 0
    if isinstance(device_queue, (list, tuple)) and device_queue:
        try:
            identity = mp.current_process()._identity
            rank = int(identity[0]) - 1 if identity else 0
            device = int(device_queue[rank % len(device_queue)])
        except Exception:
            device = int(device_queue[0])
    elif isinstance(device_queue, int):
        device = int(device_queue)

    os.environ["ALLENNLP_SRL_MODEL_PATH"] = str(model_path)
    os.environ["ALLENNLP_SRL_CUDA_DEVICE"] = str(device)
    _ALLENNLP_SRL_PREDICTOR = None
    _ALLENNLP_SRL_PREDICTOR_RAW = None
    _ALLENNLP_SRL_INIT_FAILED = False
    print(f"[worker-init] pid={os.getpid()} cuda_device={device}", flush=True)

    predictor = _get_allennlp_srl_predictor()
    if predictor is None:
        print(f"[worker-init] pid={os.getpid()} cuda_device={device} predictor_init=failed", flush=True)
        return

    try:
        predictor("a person is walking")
        print(f"[worker-init] pid={os.getpid()} cuda_device={device} predictor_init=ready", flush=True)
    except Exception:
        print(f"[worker-init] pid={os.getpid()} cuda_device={device} predictor_warmup=failed", flush=True)


def _offline_worker_build_caption(caption):
    b = build_text_branches(caption)
    return (
        caption,
        b["entity_text"],
        b["action_text"],
        int(b["entity_fallback"]),
        int(b["action_fallback"]),
    )


def _offline_worker_build_caption_batch(captions):
    if not isinstance(captions, list):
        captions = list(captions)
    if len(captions) == 0:
        return []

    srl_outputs = _predict_srl_batch(captions)
    out = []
    for caption, srl_out in zip(captions, srl_outputs):
        b = build_text_branches(caption, srl_predictor=(lambda _x, _out=srl_out: _out))
        out.append((
            caption,
            b["entity_text"],
            b["action_text"],
            int(b["entity_fallback"]),
            int(b["action_fallback"]),
        ))
    return out


def _resolve_allennlp_model_path_once(model_path):
    model_path = _normalize_space(model_path)
    if not model_path:
        return model_path

    if not re.match(r"^https?://", model_path.lower()):
        return model_path

    try:
        from allennlp.common.file_utils import cached_path

        local_path = cached_path(model_path)
        return str(local_path)
    except Exception:
        return model_path


def _extract_entities_from_srl_frames(frames):
    entities = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for role, text in frame.items():
            rr = _normalize_role(role)
            if rr.startswith("ARGM") or rr in {"V", "VERB"}:
                continue
            if not re.match(r"^ARG\d+$", rr):
                continue
            cleaned = _normalize_argument_tokens(_tokenize(text))
            cleaned = _clean_phrase_tokens(cleaned)
            if cleaned and any(_is_candidate_noun_token(t) for t in cleaned):
                entities.extend(_entity_chunks_from_tokens(cleaned))
    return _dedup_phrases(entities)


def build_text_branches(caption, srl_predictor=None):
    caption = _normalize_space(caption)

    predictor = srl_predictor if srl_predictor is not None else _get_allennlp_srl_predictor()
    srl_output = None
    if predictor is not None:
        try:
            srl_output = predictor(caption)
        except Exception:
            srl_output = None

    srl_frames = _collect_srl_frames(srl_output)

    entity_phrases = _extract_entities_from_srl_frames(srl_frames)
    if not entity_phrases:
        entity_phrases = _extract_entity_phrases_rule(caption)

    if entity_phrases:
        entity_text = _join_phrases_with_and(entity_phrases)
        entity_fallback = 0
    else:
        entity_text = caption
        entity_fallback = 1

    # SRL-first and SRL-core-only action phrase generation.
    action_phrases = _compose_semantic_phrases_from_frames(srl_frames)

    # keep concise and remove near-duplicates by prefix
    compact = []
    for ph in action_phrases:
        key = ph.lower()
        if any(key == p.lower() or key.startswith(p.lower() + " ") for p in compact):
            continue
        compact.append(ph)

    pruned = []
    for i, ph in enumerate(compact):
        toks_i = _tokenize(ph)
        drop = False
        for j, other in enumerate(compact):
            if i == j:
                continue
            toks_j = _tokenize(other)
            if len(toks_j) <= len(toks_i):
                continue
            if len(toks_i) <= 4 and _is_subsequence(toks_i, toks_j):
                drop = True
                break
        if not drop:
            pruned.append(ph)
    compact = pruned
    compact = [_trim_optional_tail_pp(p) for p in compact]
    compact = _dedup_phrases(compact)

    if compact:
        action_text = _join_phrases_with_and(compact)
        action_fallback = 0
    else:
        action_text = caption
        action_fallback = 1

    return {
        "original_text": caption,
        "entity_text": entity_text,
        "action_text": action_text,
        "entity_fallback": entity_fallback,
        "action_fallback": action_fallback,
    }


def load_text_branch_cache(cache_path="", default_path="", required=True, cache_name="text_branch_cache"):
    branch_cache = {}
    use_path = ""
    if isinstance(cache_path, str) and cache_path.strip():
        use_path = cache_path.strip()
    elif isinstance(default_path, str) and default_path.strip():
        use_path = default_path.strip()

    if not use_path:
        if required:
            raise FileNotFoundError("{} path is empty; offline cache is required.".format(cache_name))
        return branch_cache

    if not os.path.exists(use_path):
        if required:
            raise FileNotFoundError("{} not found: {}".format(cache_name, use_path))
        return branch_cache

    try:
        with open(use_path, "r", encoding="utf-8") as f:
            cache_obj = json.load(f)
        if isinstance(cache_obj, dict) and isinstance(cache_obj.get("branches"), dict):
            branch_cache = cache_obj["branches"]
        elif isinstance(cache_obj, dict):
            branch_cache = cache_obj
    except Exception as ex:
        if required:
            raise RuntimeError("Failed to load {} {}: {}".format(cache_name, use_path, ex))
        print("Failed to load text branch cache {}: {}".format(use_path, ex))
        return {}

    if required and len(branch_cache) == 0:
        raise RuntimeError("{} is empty: {}".format(cache_name, use_path))

    print("Loaded text branch cache: {} ({} entries)".format(use_path, len(branch_cache)))
    return branch_cache


def get_text_branches_from_cache_or_build(caption, branch_cache=None, require_cache=True):
    if isinstance(branch_cache, dict):
        rec = branch_cache.get(caption)
        if isinstance(rec, dict):
            entity_text = _normalize_space(rec.get("entity_text", ""))
            action_text = _normalize_space(rec.get("action_text", ""))
            entity_fallback = int(rec.get("entity_fallback", 1))
            action_fallback = int(rec.get("action_fallback", 1))

            if entity_text and action_text:
                return {
                    "original_text": _normalize_space(caption),
                    "entity_text": entity_text,
                    "action_text": action_text,
                    "entity_fallback": entity_fallback,
                    "action_fallback": action_fallback,
                }
            if require_cache:
                raise KeyError("Offline cache record is incomplete for caption: '{}'".format(caption[:120]))

    if require_cache:
        raise KeyError("Caption not found in offline cache: '{}'".format(caption[:120]))

    return build_text_branches(caption)


def load_text_branch_records(cache_path="", default_path="", required=True, cache_name="text_branch_records"):
    use_path = ""
    if isinstance(cache_path, str) and cache_path.strip():
        use_path = cache_path.strip()
    elif isinstance(default_path, str) and default_path.strip():
        use_path = default_path.strip()

    if not use_path:
        if required:
            raise FileNotFoundError("{} path is empty; offline cache is required.".format(cache_name))
        return []

    if not os.path.exists(use_path):
        if required:
            raise FileNotFoundError("{} not found: {}".format(cache_name, use_path))
        return []

    try:
        with open(use_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as ex:
        if required:
            raise RuntimeError("Failed to load {} {}: {}".format(cache_name, use_path, ex))
        return []

    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        if required:
            raise RuntimeError("{} missing ordered 'records' list: {}".format(cache_name, use_path))
        return []

    return payload["records"]


def get_text_branches_from_records(records, index, caption, require_match=True):
    if not isinstance(records, list) or index < 0 or index >= len(records):
        raise IndexError("records index out of range: {}".format(index))

    rec = records[index]
    if not isinstance(rec, dict):
        raise RuntimeError("Invalid record type at index {}".format(index))

    rec_caption = _normalize_space(rec.get("caption", ""))
    cur_caption = _normalize_space(caption)
    if require_match and rec_caption != cur_caption:
        raise RuntimeError(
            "Caption mismatch at index {}: '{}' != '{}'".format(index, rec_caption[:120], cur_caption[:120])
        )

    entity_text = _normalize_space(rec.get("entity_text", ""))
    action_text = _normalize_space(rec.get("action_text", ""))
    if not entity_text or not action_text:
        raise RuntimeError("Incomplete entity/action text at index {}".format(index))

    return {
        "original_text": cur_caption,
        "entity_text": entity_text,
        "action_text": action_text,
        "entity_fallback": int(rec.get("entity_fallback", 1)),
        "action_fallback": int(rec.get("action_fallback", 1)),
    }


def _cli_parse_args():
    parser = argparse.ArgumentParser("Build offline MSVD text branch cache")
    parser.add_argument("--data_path", type=str, required=True, help="MSVD root path containing msvd_train/val/test.json")
    parser.add_argument(
        "--allennlp_srl_model_path",
        type=str,
        default=_DEFAULT_ALLENNLP_SRL_MODEL,
        help="AllenNLP SRL model path or URL",
    )
    parser.add_argument(
        "--allennlp_srl_cuda_device",
        type=str,
        default="0,1,6,7",
        help="AllenNLP cuda devices for offline build, e.g. 0,1,6,7 or -1",
    )
    parser.add_argument("--subsets", type=str, default="train,val,test", help="Comma separated subsets")
    parser.add_argument("--output_dir", type=str, default=os.path.join(_ROOT_DIR, "ckpts", "msvd_eval", "text_branches"), help="Output directory inside workspace")
    parser.add_argument("--output_suffix", type=str, default="_text_branches.json", help="Output file suffix appended after msvd_<subset>")
    return parser.parse_args()


def _build_offline_cache_cli(args):
    resolved_model_path = _resolve_allennlp_model_path_once(args.allennlp_srl_model_path)
    os.environ["ALLENNLP_SRL_MODEL_PATH"] = resolved_model_path
    selected_devices = _resolve_offline_cuda_devices(args.allennlp_srl_cuda_device)
    os.environ["ALLENNLP_SRL_CUDA_DEVICE"] = str(selected_devices[0])
    print(
        "[info] offline_srl_devices="
        + ",".join(str(x) for x in selected_devices)
        + f", CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )
    print(
        "[info] allennlp_srl_model_path="
        + str(resolved_model_path)
        + (" (cached_local)" if resolved_model_path != args.allennlp_srl_model_path else "")
    )

    output_dir = os.path.abspath(args.output_dir)
    root_dir = os.path.abspath(_ROOT_DIR)
    if os.path.commonpath([output_dir, root_dir]) != root_dir:
        raise ValueError(f"output_dir must be inside workspace: {root_dir}, got {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    subsets = [x.strip() for x in str(args.subsets).split(",") if x.strip()]
    for subset in subsets:
        in_file = os.path.join(args.data_path, f"msvd_{subset}.json")
        out_file = os.path.join(output_dir, f"msvd_{subset}{args.output_suffix}")

        if not os.path.exists(in_file):
            print(f"[skip] missing input: {in_file}")
            continue

        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_captions = []
        for item in data:
            caps = item.get("caption", [])
            if isinstance(caps, str):
                caps = [caps]
            all_captions.extend([c.strip() for c in caps if isinstance(c, str) and c.strip()])

        uniq = sorted(set(all_captions))
        print(f"[info] subset={subset}, unique_captions={len(uniq)}")
        log_every = 200
        branches = {}
        ef = 0
        af = 0
        if selected_devices == [-1] or len(selected_devices) <= 1 or len(uniq) < 256:
            iterator = uniq
            if tqdm is not None:
                iterator = tqdm(uniq, total=len(uniq), desc=f"[{subset}]", dynamic_ncols=True)
            for i, cap in enumerate(iterator, start=1):
                b = build_text_branches(cap)
                branches[cap] = {
                    "entity_text": b["entity_text"],
                    "action_text": b["action_text"],
                    "entity_fallback": int(b["entity_fallback"]),
                    "action_fallback": int(b["action_fallback"]),
                }
                ef += int(b["entity_fallback"])
                af += int(b["action_fallback"])
                if tqdm is None and i % log_every == 0:
                    print(f"[{subset}] {i}/{len(uniq)}")
        else:
            ctx = mp.get_context("spawn")
            workers = len(selected_devices)
            chunksize = max(16, len(uniq) // (workers * 32))
            print(f"[info] subset={subset}, mode=multi-gpu, workers={workers}, chunksize={chunksize}")
            with ctx.Pool(
                processes=workers,
                initializer=_offline_worker_init,
                initargs=(resolved_model_path, tuple(int(d) for d in selected_devices)),
            ) as pool:
                stream = pool.imap_unordered(_offline_worker_build_caption, uniq, chunksize=chunksize)
                progress = None
                if tqdm is not None:
                    progress = tqdm(total=len(uniq), desc=f"[{subset}]", dynamic_ncols=True)

                for i, (cap, entity_text, action_text, entity_fb, action_fb) in enumerate(stream, start=1):
                    branches[cap] = {
                        "entity_text": entity_text,
                        "action_text": action_text,
                        "entity_fallback": int(entity_fb),
                        "action_fallback": int(action_fb),
                    }
                    ef += int(entity_fb)
                    af += int(action_fb)
                    if progress is not None:
                        progress.update(1)
                    elif i % log_every == 0:
                        print(f"[{subset}] {i}/{len(uniq)}")

                if progress is not None:
                    progress.close()

        payload = {
            "meta": {
                "subset": subset,
                "num_unique_captions": len(uniq),
                "srl_backend": "allennlp",
                "allennlp_srl_model_path": args.allennlp_srl_model_path,
                "allennlp_srl_cuda_device": ",".join(str(x) for x in selected_devices),
                "output_dir": output_dir,
                "source_file": "dataloaders/text_branch_utils.py",
            },
            "branches": branches,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        n = max(len(uniq), 1)
        print(f"[done] {subset}: saved -> {out_file}")
        print(f"        unique={len(uniq)}, entity_fallback={ef / n:.4f}, action_fallback={af / n:.4f}")


if __name__ == "__main__":
    _build_offline_cache_cli(_cli_parse_args())
