from typing import Optional, Tuple

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    
    # -------------------------------
    # (4) Flash fast-path detection
    # -------------------------------
    # Conditions where we can safely drop the mask and use is_causal=True:
    #  - self-attention (Q,K,V same sequence length)
    #  - batch size == 1 (no cross-example padding)
    #  - no *real* padding present; masks seen are either:
    #      * None
    #      * all-True boolean mask
    #      * square float additive mask (e.g., causal upper-tri with -inf), which we can replace
    B = query.shape[0]
    Lq = query.shape[-2]; Lk = key.shape[-2]; Lv = value.shape[-2]
    same_len = (Lq == Lk == Lv)

    def _is_trivial_bool_mask(t):
        return (
            t is not None
            and torch.is_tensor(t)
            and t.dtype == torch.bool
            and t.ndim >= 2
            and bool(t.all().item())
        )

    def _looks_like_additive_square_mask(t):
        # float mask with shape [..., L, L]; typical for additive causal/padding masks
        return (
            t is not None
            and torch.is_tensor(t)
            and t.dtype.is_floating_point
            and t.shape[-1] == t.shape[-2]
        )

    no_pad_and_causal_ok = (
        B == 1
        and same_len
        and (
            causal_mask is None
            or _is_trivial_bool_mask(causal_mask)
            or _looks_like_additive_square_mask(causal_mask)
        )
    )

    # If conditions are met, drop the mask and set is_causal=True
    attn_mask_arg = causal_mask
    is_causal_arg = is_causal
    if no_pad_and_causal_ok:
        attn_mask_arg = None
        is_causal_arg = True  # lets SDPA/Flash build the causal mask internally

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask_arg,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal_arg,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
