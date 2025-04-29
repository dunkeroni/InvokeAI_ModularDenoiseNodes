import torch.nn as nn
import torch
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0
import torch.nn.functional as F
from torchvision.transforms.functional import resize as tv_resize
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from typing import List, Optional, cast
import math

import sys
import os
import importlib.util

# Assuming submodule is at "nodes/InvokeAI_ModularDenoiseNodes/entmax/entmax"
submodule_path = os.path.join(os.path.dirname(__file__), "entmax")
if submodule_path not in sys.path:
    print(f"Adding {submodule_path} to sys.path")
    sys.path.insert(0, submodule_path)

# Now you can import the entmax module as expected.
from entmax import sparsemax, entmax15

# torch description of F.scaled_dot_product_attention:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def sparsemax_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = sparsemax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def entmax15_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = entmax15(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class EntmaxAttention(CustomAttnProcessor2_0):
    @torch.no_grad()
    def __init__(self, alpha: float, scale: float, *args, **kwargs):
        self.alpha = alpha
        self.scale = scale
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def override_attention(self, *args, **kwargs):
        sdp_result = F.scaled_dot_product_attention(*args, **kwargs)
        if not self.is_cross_attention:
            # use the default softmax attention
            return sdp_result

        if self.alpha == 1.5:
            # use the entmax15 attention
            entmax_result = entmax15_attention(*args, **kwargs)
        else: #self.alpha == 2:
            # use the sparsemax attention
            entmax_result = sparsemax_attention(*args, **kwargs)
        
        return sdp_result + self.scale * (entmax_result - sdp_result)


    @torch.no_grad()
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # For Regional Prompting:
        regional_prompt_data: Optional[RegionalPromptData] = None,
        percent_through: Optional[torch.Tensor] = None,
        # For IP-Adapter:
        regional_ip_data: Optional[RegionalIPData] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """Apply attention.
        Args:
            regional_prompt_data: The regional prompt data for the current batch. If not None, this will be used to
                apply regional prompt masking.
            regional_ip_data: The IP-Adapter data for the current batch.
        """
        # If true, we are doing cross-attention, if false we are doing self-attention.
        self.is_cross_attention = encoder_hidden_states is not None

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End unmodified block from AttnProcessor2_0.

        _, query_seq_len, _ = hidden_states.shape
        # Handle regional prompt attention masks.
        if regional_prompt_data is not None and self.is_cross_attention:
            assert percent_through is not None
            prompt_region_attention_mask = regional_prompt_data.get_cross_attn_mask(
                query_seq_len=query_seq_len, key_seq_len=sequence_length
            )

            if attention_mask is None:
                attention_mask = prompt_region_attention_mask
            else:
                attention_mask = prompt_region_attention_mask + attention_mask

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = self.override_attention( #F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End unmodified block from AttnProcessor2_0.

        # Apply IP-Adapter conditioning.
        if self.is_cross_attention:
            if self._ip_adapter_attention_weights:
                assert regional_ip_data is not None
                ip_masks = regional_ip_data.get_masks(query_seq_len=query_seq_len)

                assert (
                    len(regional_ip_data.image_prompt_embeds)
                    == len(self._ip_adapter_attention_weights)
                    == len(regional_ip_data.scales)
                    == ip_masks.shape[1]
                )

                for ipa_index, ipa_embed in enumerate(regional_ip_data.image_prompt_embeds):
                    ipa_weights = self._ip_adapter_attention_weights[ipa_index].ip_adapter_weights
                    ipa_scale = regional_ip_data.scales[ipa_index]
                    ip_mask = ip_masks[0, ipa_index, ...]

                    # The batch dimensions should match.
                    assert ipa_embed.shape[0] == encoder_hidden_states.shape[0]
                    # The token_len dimensions should match.
                    assert ipa_embed.shape[-1] == encoder_hidden_states.shape[-1]

                    ip_hidden_states = ipa_embed

                    # Expected ip_hidden_state shape: (batch_size, num_ip_images, ip_seq_len, ip_image_embedding)

                    if not self._ip_adapter_attention_weights[ipa_index].skip:
                        ip_key = ipa_weights.to_k_ip(ip_hidden_states)
                        ip_value = ipa_weights.to_v_ip(ip_hidden_states)

                        # Expected ip_key and ip_value shape:
                        # (batch_size, num_ip_images, ip_seq_len, head_dim * num_heads)

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # Expected ip_key and ip_value shape:
                        # (batch_size, num_heads, num_ip_images * ip_seq_len, head_dim)

                        # TODO: add support for attn.scale when we move to Torch 2.1
                        ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        # Expected ip_hidden_states shape: (batch_size, num_heads, query_seq_len, head_dim)
                        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )

                        ip_hidden_states = ip_hidden_states.to(query.dtype)

                        # Expected ip_hidden_states shape: (batch_size, query_seq_len, num_heads * head_dim)
                        hidden_states = hidden_states + ipa_scale * ip_hidden_states * ip_mask
            else:
                # If IP-Adapter is not enabled, then regional_ip_data should not be passed in.
                assert regional_ip_data is None

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End of unmodified block from AttnProcessor2_0

        # casting torch.Tensor to torch.FloatTensor to avoid type issues
        return cast(torch.FloatTensor, hidden_states)