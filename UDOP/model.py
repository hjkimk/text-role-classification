import os
import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutput
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Block, T5ForConditionalGeneration, T5LayerNorm

from dataclasses import dataclass

from core.models import UdopUnimodelForConditionalGeneration, T52dStack
from core.models.mae.build import mae_model
from core.models.embedding.cell_embed import CellEmbeddings
from core.models.embedding.relative.relative import (
    RelativePositionBias1D,
    RelativePositionBiasAggregated,
    RelativePositionBiasBase,
    create_relative_bias,
)

from utils_config import utils_config


# adapted from original UDOP source code


def pad_sequence(seq, target_len, pad_value=0):
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq)
    m = target_len - n
    if m > 0:
        ret = torch.stack([pad_value] * m).to(seq)
        seq = torch.cat([seq, ret], dim=0)
    return seq[:target_len]


def collate_vlembed(inputs_patches, inputs_embeds, seg_data, visual_segdata, vis_special_token=None,
                    attention_mask=None, num_patches=14, max_len=0):
    L = num_patches
    ocr_points_x = torch.clip(torch.floor((seg_data[:, :, 0] + seg_data[:, :, 2]) / 2.0 * L).long(), 0, L - 1)
    ocr_points_y = torch.clip(torch.floor((seg_data[:, :, 1] + seg_data[:, :, 3]) / 2.0 * L).long(), 0, L - 1) * L
    ocr_points = ocr_points_x + ocr_points_y
    target_seg = (seg_data.mean(-1) == 0.0) | (seg_data.mean(-1) == 1.0)
    repeated_vision_embeds = torch.gather(inputs_patches, 1,
                                          ocr_points.unsqueeze(-1).repeat(1, 1, inputs_patches.size(-1)))
    repeated_vision_embeds[target_seg] = 0.0
    inputs_embeds += repeated_vision_embeds

    patch_inds = torch.full_like(inputs_patches[:, :, 0], True).bool()
    ind = torch.cat([torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points),
                     ocr_points[:, :, None]], -1).flatten(0, 1)
    rows, cols = zip(*ind)
    patch_inds[rows, cols] = False

    input_vision_patches = [inputs_patches[i][patch_inds[i]] for i in range(len(patch_inds))]
    visual_segdata = [visual_segdata[i][patch_inds[i]] for i in range(len(patch_inds))]
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_segdata]

    if max_len == 0:
        max_len = inputs_patches.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)
    inputs_vision_patches = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(inputs_patches[0, 0])) for item in input_vision_patches])
    visual_segdata = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(seg_data[0, 0])) for item in visual_segdata])
    if attention_mask is not None:
        visual_attention_mask = torch.stack(
            [pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask])

    if vis_special_token is not None:
        inputs_vision_patches += vis_special_token

    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    seg_data = torch.cat([seg_data, visual_segdata], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)

    return inputs_embeds, seg_data, attention_mask


@dataclass
class BaseModelOutputWithVisionEmbeds(BaseModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_embeds: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    seg_data: torch.FloatTensor = None


class T52dStack(T5PreTrainedModel):
    """
    Almost exact copy of transformers T5Stack with the modification
    of passing `position_bias` in the forward method
    """

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length

        setattr(config, 'output_attentions', True)
        if self.is_decoder:
            self.num_layers = (
                config.truncate_decoder_after_layer if config.truncate_decoder_after_layer else config.num_layers
            )
        else:
            self.num_layers = (
                config.truncate_encoder_after_layer if config.truncate_encoder_after_layer else config.num_layers
            )

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        if not self.is_decoder:
            self.cell2dembedding = CellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)

        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )

        self.init_weights()

    @staticmethod
    def _get_relative_bias(config: T5Config) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            ids_keep=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cross_attn_head_mask=None,
            position_bias=None,  # modified line,
            inputs_patches=None,  # modified line,
            seg_data=None,  # modified line,
            visual_seg_data=None,  # modified line,
            num_patches=None,  # modified line,
            special_vis_token=None,  # modified line,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = True  # False #True #output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ======================================================
        # input embeddings processing

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            seg_data = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(
                self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
            )
            # encoder_attention_mask = attention_mask
            # logger.warning('Empty batch')
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if inputs_patches is not None:
            # ===========================
            # combine OCR text and visual embed
            inputs_embeds, seg_data, attention_mask = collate_vlembed(inputs_patches, inputs_embeds, seg_data,
                                                                      visual_seg_data, special_vis_token,
                                                                      attention_mask, num_patches, 0)
            input_shape = inputs_embeds.size()[:-1]

        if not self.is_decoder:
            inputs_embeds += self.cell2dembedding(seg_data)

        batch_size, seq_length = input_shape

        # ======================================================
        # input masking/pos embed processing

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(self)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = self.relative_bias(
                attention_mask=attention_mask, seg_data=seg_data
            )
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None

        # ======================================================
        # model inferencing

        hidden_states = inputs_embeds

        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:  # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithVisionEmbeds(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask,
            seg_data=seg_data,
        )


class UdopUniModelClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.d_model * 3, config.d_model)
        else:
            self.dense = nn.Linear(config.d_model, config.d_model)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class UdopUnimodelForTokenClassification(UdopUnimodelForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config, classifier):
        super().__init__(config)
        self.model_dim = config.d_model
        self.num_labels = len(utils_config["label_list"])

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = T52dStack(encoder_config, self.shared)

        # initialize weights
        self.init_weights()

        mae_model_tmp = mae_model(config.mae_version, config.mae_checkpoint, config.image_size, config.vocab_size,
                                  config.max_2d_position_embeddings)

        self.patch_embed = mae_model_tmp.patch_embed
        self.embed_dim = mae_model_tmp.embed_dim
        self.pos_embed = mae_model_tmp.pos_embed
        self.special_vis_token = mae_model_tmp.special_vis_token

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if classifier == "linear":
            self.classifier = nn.Linear(config.d_model, self.num_labels)
        else:
            self.classifier = UdopUniModelClassificationHead(config, pool_feature=False)
        self.weights = None  # modified

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Tensor = None,
            inputs_embeds: Optional[Tensor] = None,
            encoder_outputs: Optional[Tensor] = None,
            image: Optional[Tensor] = None,
            seg_data: Dict[str, Any] = None,
            visual_seg_data: Dict[str, Any] = None,
            labels: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            inputs_patches = None
            if image is not None:
                assert visual_seg_data is not None
                x = self.patch_embed(image)
                num_patches = image.size(2) // 16
                inputs_patches = x

        outputs = self.encoder(
            input_ids=input_ids,
            seg_data=seg_data,
            visual_seg_data=visual_seg_data,
            inputs_patches=inputs_patches,
            num_patches=num_patches,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        sequence_length = input_shape[1]
        attentions = []

        # only take text part of output representations
        sequence_output = outputs[0][:, :sequence_length]
        hidden_states = outputs.last_hidden_state[:, :sequence_length]
        for layer in outputs.attentions:
            layer = layer[:, :, :sequence_length, :sequence_length]
            attentions.append(layer)
        attentions = tuple(attentions)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.weights is not None:
                self.weights = self.weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=self.weights)  # modified
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (active_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions
        )
