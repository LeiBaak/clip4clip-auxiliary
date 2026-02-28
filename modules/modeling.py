from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ProST.decoder import Event_decoder, Frame_decoder

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()

        self.use_entity_branch = True
        self.use_action_branch = True
        self.lambda_entity = 0.3
        self.lambda_action = 0.3
        if hasattr(task_config, "use_entity_branch"):
            self.use_entity_branch = task_config.use_entity_branch
        if hasattr(task_config, "use_action_branch"):
            self.use_action_branch = task_config.use_action_branch
        if hasattr(task_config, "lambda_entity"):
            self.lambda_entity = float(task_config.lambda_entity)
        if hasattr(task_config, "lambda_action"):
            self.lambda_action = float(task_config.lambda_action)

        self.entity_word_pro_num = 28
        self.entity_patch_num = 12
        self.use_entity_query_attention = True
        if hasattr(task_config, "entity_word_pro_num"):
            self.entity_word_pro_num = int(task_config.entity_word_pro_num)
        if hasattr(task_config, "entity_patch_num"):
            self.entity_patch_num = int(task_config.entity_patch_num)
        if hasattr(task_config, "use_entity_query_attention"):
            self.use_entity_query_attention = bool(task_config.use_entity_query_attention)
        show_log(task_config, "\t use_entity_query_attention(patch-proto): {}".format(self.use_entity_query_attention))
        if self.entity_patch_num < 2:
            self.entity_patch_num = 2

        self.entity_word_prototype_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, self.entity_word_pro_num),
            nn.ReLU(inplace=True),
        )
        self.entity_patch_prototype_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, max(self.entity_patch_num - 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.entity_event_layer_num = int(getattr(task_config, "event_layer_num", 1))
        self.entity_max_vfea = int(getattr(task_config, "max_vfea", self.task_config.max_frames))
        self.entity_frame_decoder = Frame_decoder(
            num_attris=self.task_config.max_frames,
            layers=self.entity_event_layer_num,
            heads=1,
            dim_ftr=embed_dim,
            pos_emb=False,
            length=1,
            dim_feedforward=embed_dim,
            without_init=False,
        )
        self.entity_event_decoder = Event_decoder(
            num_attris=self.entity_max_vfea,
            layers=self.entity_event_layer_num,
            heads=1,
            dim_ftr=embed_dim,
            pos_emb=False,
            length=1,
            dim_feedforward=embed_dim,
            without_init=False,
        )

        self.apply(self.init_weights)

    def _safe_logit_scale(self):
        logit_scale_param = torch.clamp(self.clip.logit_scale, max=math.log(100.0))
        return logit_scale_param.exp()

    def _safe_l2_normalize(self, tensor, dim=-1, eps=1e-6):
        denom = torch.clamp(tensor.norm(dim=dim, keepdim=True), min=eps)
        return tensor / denom

    def _mean_pool_text_embed(self, sequence_output, attention_mask=None):
        """
        Input:
            sequence_output: [B, 1, D] or [B, L, D]
            attention_mask: [B, L] or None
        Output:
            text_embed: [B, D]
        """
        if sequence_output.dim() == 3 and sequence_output.size(1) == 1:
            text_embed = sequence_output.squeeze(1)
        else:
            if attention_mask is None:
                text_embed = sequence_output.mean(dim=1)
            else:
                attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
                denom = torch.sum(attention_mask_un, dim=1, dtype=torch.float)
                denom[denom == 0.] = 1.
                text_embed = torch.sum(sequence_output * attention_mask_un, dim=1) / denom
        text_embed = self._safe_l2_normalize(text_embed)
        return text_embed

    def _pool_entity_video_embed(self, visual_output, video_mask):
        """
        A1 entity-video branch (simplified single vector).

        Input:
            visual_output: [B, T, D]
            video_mask: [B, T]
        Output:
            entity_video_embed: [B, D]

        Notes:
            - A1 uses mean pooling over valid frames as an entity approximation.
            - Future upgrade path: replace this with K entity prototypes from patch/token features.
        """
        entity_video_embed = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        entity_video_embed = self._safe_l2_normalize(entity_video_embed)
        return entity_video_embed

    def _pool_action_video_embed(self, visual_output, video_mask):
        """
        A1 action-video branch (simplified temporal dynamics).

        Input:
            visual_output: [B, T, D]
            video_mask: [B, T]
        Output:
            action_video_embed: [B, D]

        Fallback logic:
            - Primary: adjacent frame difference features + masked mean pooling.
            - Fallback: if no valid diff frames exist, fallback to global mean pooled visual embedding.

        Notes:
            - No tracking/tube/optical-flow in A1.
            - Future upgrade path: entity-based temporal prototype dynamics.
        """
        if visual_output.size(1) <= 1:
            fallback_embed = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
            return self._safe_l2_normalize(fallback_embed)

        diff_feat = visual_output[:, 1:, :] - visual_output[:, :-1, :]
        diff_mask = (video_mask[:, 1:] * video_mask[:, :-1]).to(dtype=torch.float)
        diff_mask_un = diff_mask.unsqueeze(-1)

        diff_sum = torch.sum(diff_feat * diff_mask_un, dim=1)
        diff_cnt = torch.sum(diff_mask_un, dim=1, dtype=torch.float)

        fallback_embed = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        action_video_embed = diff_sum / torch.clamp(diff_cnt, min=1.0)

        valid = (diff_cnt.squeeze(-1) > 0).to(dtype=action_video_embed.dtype).unsqueeze(-1)
        action_video_embed = action_video_embed * valid + fallback_embed * (1.0 - valid)
        action_video_embed = self._safe_l2_normalize(action_video_embed)
        return action_video_embed

    def _branch_similarity_logits(self, text_embed, video_embed):
        """
        Input:
            text_embed: [B_t, D]
            video_embed: [B_v, D]
        Output:
            logits: [B_t, B_v]
        """
        if self.training:
            text_embed = allgather(text_embed, self.task_config)
            video_embed = allgather(video_embed, self.task_config)
            torch.distributed.barrier()

        logit_scale = self._safe_logit_scale()
        return logit_scale * torch.matmul(text_embed, video_embed.t())

    def get_entity_text_word_prototypes(self, input_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        _, hidden = self.clip.encode_text(input_ids, return_hidden=True)
        hidden = hidden.float()

        word_weights = self.entity_word_prototype_weight(hidden)
        text_word_proto = torch.einsum("bld,blp->bpd", hidden, word_weights)
        return text_word_proto

    def get_entity_video_patch_prototypes(self, visual_tokens):
        cls_tokens = visual_tokens[:, :, 0, :]

        bsz, frame_len, token_len, feat_dim = visual_tokens.shape
        tokens_flat = visual_tokens.reshape(bsz * frame_len, token_len, feat_dim)

        patch_weights = self.entity_patch_prototype_weight(tokens_flat)
        patch_proto = torch.einsum("bnd,bnk->bkd", tokens_flat, patch_weights)

        patch_proto = patch_proto.reshape(bsz, frame_len, -1, feat_dim)
        frame_proto = torch.cat((cls_tokens.unsqueeze(2), patch_proto), dim=2)
        return frame_proto

    def _entity_prototype_similarity_logits(self, text_word_proto, video_patch_proto, video_mask):
        text_word_proto = text_word_proto.contiguous()
        video_patch_proto = video_patch_proto.contiguous()
        video_mask = video_mask.contiguous()
        if self.training:
            text_word_proto = allgather(text_word_proto, self.task_config)
            video_patch_proto = allgather(video_patch_proto, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            torch.distributed.barrier()

        text_word_proto = F.normalize(text_word_proto, p=2, dim=2)
        video_patch_proto = F.normalize(video_patch_proto, p=2, dim=3)

        sim = torch.einsum("aid,bfjd->abfij", text_word_proto, video_patch_proto)
        sim = sim.max(dim=3)[0]
        sim = sim.max(dim=2)[0]
        logits = sim.sum(dim=-1) / max(float(self.entity_patch_num), 1.0)

        logit_scale = self._safe_logit_scale()
        return logit_scale * logits

    def get_multi_branch_similarity_logits(
            self,
            sequence_output_global,
            sequence_output_entity,
            sequence_output_action,
            visual_output,
            attention_mask_global,
            attention_mask_entity,
            attention_mask_action,
            video_mask,
            entity_word_proto=None,
            entity_video_patch_proto=None,
            loose_type=False,
    ):
        """
        Input:
            sequence_output_global/entity/action: [B, 1, D]
            visual_output: [B, T, D]
            attention_mask_*: [B, L]
            video_mask: [B, T]
        Output:
            logits_global, logits_entity, logits_action: each [B, B]
        """
        logits_global, *_ = self.get_similarity_logits(
            sequence_output_global,
            visual_output,
            attention_mask_global,
            video_mask,
            shaped=True,
            loose_type=loose_type,
        )

        text_entity_embed = self._mean_pool_text_embed(sequence_output_entity, attention_mask_entity)
        text_action_embed = self._mean_pool_text_embed(sequence_output_action, attention_mask_action)
        video_action_embed = self._pool_action_video_embed(visual_output, video_mask)

        if entity_word_proto is not None and entity_video_patch_proto is not None:
            logits_entity = self._entity_prototype_similarity_logits(entity_word_proto, entity_video_patch_proto, video_mask)
        else:
            video_entity_embed = self._pool_entity_video_embed(visual_output, video_mask)
            logits_entity = self._branch_similarity_logits(text_entity_embed, video_entity_embed)
        logits_action = self._branch_similarity_logits(text_action_embed, video_action_embed)
        return logits_global, logits_entity, logits_action

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask,
            video,
            video_mask=None,
            entity_input_ids=None,
            entity_token_type_ids=None,
            entity_attention_mask=None,
            action_input_ids=None,
            action_token_type_ids=None,
            action_attention_mask=None,
    ):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        if entity_input_ids is None:
            entity_input_ids = input_ids
            entity_token_type_ids = token_type_ids
            entity_attention_mask = attention_mask
        else:
            entity_input_ids = entity_input_ids.view(-1, entity_input_ids.shape[-1])
            entity_token_type_ids = entity_token_type_ids.view(-1, entity_token_type_ids.shape[-1])
            entity_attention_mask = entity_attention_mask.view(-1, entity_attention_mask.shape[-1])

        if action_input_ids is None:
            action_input_ids = input_ids
            action_token_type_ids = token_type_ids
            action_attention_mask = attention_mask
        else:
            action_input_ids = action_input_ids.view(-1, action_input_ids.shape[-1])
            action_token_type_ids = action_token_type_ids.view(-1, action_token_type_ids.shape[-1])
            action_attention_mask = action_attention_mask.view(-1, action_attention_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, visual_output, visual_tokens = self.get_sequence_visual_output(
            input_ids,
            token_type_ids,
            attention_mask,
            video,
            video_mask,
            shaped=True,
            video_frame=video_frame,
            return_hidden_visual=True,
        )
        entity_sequence_output = self.get_sequence_output(
            entity_input_ids,
            entity_token_type_ids,
            entity_attention_mask,
            shaped=True,
        )
        action_sequence_output = self.get_sequence_output(
            action_input_ids,
            action_token_type_ids,
            action_attention_mask,
            shaped=True,
        )

        entity_word_proto = self.get_entity_text_word_prototypes(
            entity_input_ids,
            entity_attention_mask,
            shaped=True,
        )
        entity_video_patch_proto = self.get_entity_video_patch_prototypes(visual_tokens)

        if self.training:
            logits_global, logits_entity, logits_action = self.get_multi_branch_similarity_logits(
                sequence_output,
                entity_sequence_output,
                action_sequence_output,
                visual_output,
                attention_mask,
                entity_attention_mask,
                action_attention_mask,
                video_mask,
                entity_word_proto=entity_word_proto,
                entity_video_patch_proto=entity_video_patch_proto,
                loose_type=self.loose_type,
            )

            loss_global = (self.loss_fct(logits_global) + self.loss_fct(logits_global.T)) / 2

            if self.use_entity_branch:
                loss_entity = (self.loss_fct(logits_entity) + self.loss_fct(logits_entity.T)) / 2
            else:
                loss_entity = torch.zeros_like(loss_global)

            if self.use_action_branch:
                loss_action = (self.loss_fct(logits_action) + self.loss_fct(logits_action.T)) / 2
            else:
                loss_action = torch.zeros_like(loss_global)

            total_loss = loss_global + self.lambda_entity * loss_entity + self.lambda_action * loss_action
            return {
                "loss": total_loss,
                "loss_global": loss_global.detach(),
                "loss_entity": loss_entity.detach(),
                "loss_action": loss_action.detach(),
                "score_global_mean": logits_global.mean().detach(),
                "score_entity_mean": logits_entity.mean().detach(),
                "score_action_mean": logits_action.mean().detach(),
            }
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1, return_hidden_visual=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_cls, visual_hidden = self.clip.encode_image(video, return_hidden=True, video_frame=video_frame)
        visual_cls = visual_cls.float()
        visual_hidden = visual_hidden.float()

        visual_output = visual_cls.view(bs_pair, -1, visual_cls.size(-1))
        if not return_hidden_visual:
            return visual_output

        token_num = visual_hidden.size(1)
        visual_tokens = visual_hidden.view(bs_pair, -1, token_num, visual_hidden.size(-1))
        return visual_output, visual_tokens

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1, return_hidden_visual=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_ret = self.get_visual_output(
            video,
            video_mask,
            shaped=True,
            video_frame=video_frame,
            return_hidden_visual=return_hidden_visual,
        )
        if return_hidden_visual:
            visual_output, visual_tokens = visual_ret
            return sequence_output, visual_output, visual_tokens

        visual_output = visual_ret
        return sequence_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = self._safe_l2_normalize(visual_output)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = self._safe_l2_normalize(visual_output)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = self._safe_l2_normalize(sequence_output)

        logit_scale = self._safe_logit_scale()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction
