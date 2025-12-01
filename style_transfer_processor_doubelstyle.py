import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,Tuple,Callable, List, Optional, Tuple, Union
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
from diffusers.utils.import_utils import  logging,is_torch_xla_available,is_torch_xla_version,is_xformers_available
from diffusers.models.attention_processor import (SpatialNorm, AttnProcessor, AttnProcessor2_0,XLAFluxFlashAttnProcessor2_0,XLAFlashAttnProcessor2_0,AttnProcessorNPU,XFormersAttnProcessor,SlicedAttnProcessor,
                                                  AttnAddedKVProcessor, AttnAddedKVProcessor2_0,
                                                  SlicedAttnAddedKVProcessor, XFormersAttnAddedKVProcessor,
                                                  CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor,
                                                  CustomDiffusionAttnProcessor2_0,
                                                  IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0,
                                                  IPAdapterXFormersAttnProcessor,
                                                  JointAttnProcessor2_0, XFormersJointAttnProcessor)

logger = logging.get_logger(__name__) 
# # 这三个函数是 Qwen-Image-Edit 的核心，你需要确保能导入它们
# # (下面是假设的导入路径，请替换为你项目中的实际路径)
# try:
#     from diffusers.models.qwen2.modeling_qwen2_vl import (
#         dispatch_attention_fn, 
#         apply_rotary_emb_qwen
#     )
# except ImportError:
#     print("Warning: Could not import Qwen functions. Using placeholders.")
#     # 定义占位符以便代码能被解析
#     def dispatch_attention_fn(*args, **kwargs):
#         return F.scaled_dot_product_attention(*args, **kwargs)
#     def apply_rotary_emb_qwen(x, *args, **kwargs):
#         return x
# # -----------------------------------------------------------------
@maybe_allow_in_graph
class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        kv_heads (`int`,  *optional*, defaults to `None`):
            The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
            `kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
            Query Attention (MQA) otherwise GQA is used.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        # To prevent circular import.
        from diffusers.models.normalization import FP32LayerNorm, LpNorm, RMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applies qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm": #架构图中的QK-Norm
            self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim_head * heads, eps=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "l2":
            self.norm_q = LpNorm(p=2, dim=-1, eps=eps)
            self.norm_k = LpNorm(p=2, dim=-1, eps=eps)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."
            )

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        # 架构图中的linear层
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "layer_norm":
                self.norm_added_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
                self.norm_added_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            elif qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
            elif qk_norm == "rms_norm_across_heads":
                # Wan applies qk norm across all heads
                # Wan also doesn't apply a q norm
                self.norm_added_q = None
                self.norm_added_k = RMSNorm(dim_head * kv_heads, eps=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:#不为none 用的是QwenDoubleStreamAttnProcessor2_0()
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_xla_flash_attention(
        self,
        use_xla_flash_attention: bool,
        partition_spec: Optional[Tuple[Optional[str], ...]] = None,
        is_flux=False,
    ) -> None:
        r"""
        Set whether to use xla flash attention from `torch_xla` or not.

        Args:
            use_xla_flash_attention (`bool`):
                Whether to use pallas flash attention kernel from `torch_xla` or not.
            partition_spec (`Tuple[]`, *optional*):
                Specify the partition specification if using SPMD. Otherwise None.
        """
        if use_xla_flash_attention:
            if not is_torch_xla_available:
                raise "torch_xla is not available"
            elif is_torch_xla_version("<", "2.3"):
                raise "flash attention pallas kernel is supported from torch_xla version 2.3"
            elif is_spmd() and is_torch_xla_version("<", "2.4"):
                raise "flash attention pallas kernel using SPMD is supported from torch_xla version 2.4"
            else:
                if is_flux:
                    processor = XLAFluxFlashAttnProcessor2_0(partition_spec)
                else:
                    processor = XLAFlashAttnProcessor2_0(partition_spec)
        else:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        r"""
        Set whether to use npu flash attention from `torch_npu` or not.

        """
        if use_npu_flash_attention:
            processor = AttnProcessorNPU()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor,
            (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0),
        )
        is_added_kv_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                AttnAddedKVProcessor,
                AttnAddedKVProcessor2_0,
                SlicedAttnAddedKVProcessor,
                XFormersAttnAddedKVProcessor,
            ),
        )
        is_ip_adapter = hasattr(self, "processor") and isinstance(
            self.processor,
            (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor),
        )
        is_joint_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                JointAttnProcessor2_0,
                XFormersJointAttnProcessor,
            ),
        )

        if use_memory_efficient_attention_xformers:
            if is_added_kv_processor and is_custom_diffusion:
                raise NotImplementedError(
                    f"Memory efficient attention is currently not supported for custom diffusion for attention processor type {self.processor}"
                )
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    dtype = None
                    if attention_op is not None:
                        op_fw, op_bw = attention_op
                        dtype, *_ = op_fw.SUPPORTED_DTYPES
                    q = torch.randn((1, 2, 40), device="cuda", dtype=dtype)
                    _ = xformers.ops.memory_efficient_attention(q, q, q)
                except Exception as e:
                    raise e

            if is_custom_diffusion:
                processor = CustomDiffusionXFormersAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_added_kv_processor:
                # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                # throw warning
                logger.info(
                    "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            elif is_ip_adapter:
                processor = IPAdapterXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    num_tokens=self.processor.num_tokens,
                    scale=self.processor.scale,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_ip"):
                    processor.to(
                        device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
                    )
            elif is_joint_processor:
                processor = XFormersJointAttnProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_custom_diffusion:
                attn_processor_class = (
                    CustomDiffusionAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else CustomDiffusionAttnProcessor
                )
                processor = attn_processor_class(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_ip_adapter:
                processor = IPAdapterAttnProcessor2_0(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    num_tokens=self.processor.num_tokens,
                    scale=self.processor.scale,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_ip"):
                    processor.to(
                        device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
                    )
            else:
                # set attention processor
                # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
                # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
                # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
                processor = (
                    AttnProcessor2_0()
                    if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
                    else AttnProcessor()
                )

        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        r"""
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        #print(f"cross_attention_kwargs: {cross_attention_kwargs.keys()}")

        # attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        # quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        # unused_kwargs = [
        #     k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        # ]
        # if len(unused_kwargs) > 0:
        #     logger.warning(
        #         f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
        #     )
        # cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
        #Attention 类的 forward 方法本身几乎不做任何计算。 
        #Attention.forward 被调用时，它立刻调用 self.processor（也就是 QwenDoubleStreamAttnProcessor2_0）的 forward 方法。
        return self.processor( #QwenDoubleStreamAttnProcessor2_0 现在拿到了所有数据（hidden_states, encoder_hidden_states）和所有零件（self.to_q, self.to_k, self.to_v, self.norm_q, self.norm_k 等）。
            self, #它把 self（即 Attention 类的实例本身）作为第一个参数传递给 processor。
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(
                    head_size, dim=0, output_size=attention_mask.shape[0] * head_size
                )
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(
                head_size, dim=1, output_size=attention_mask.shape[1] * head_size
            )

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @torch.no_grad()
    def fuse_projections(self, fuse=True):
        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                self.to_qkv.bias.copy_(concatenated_bias)

        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                self.to_kv.bias.copy_(concatenated_bias)

        # handle added projections for SD3 and others.
        if (
            getattr(self, "add_q_proj", None) is not None
            and getattr(self, "add_k_proj", None) is not None
            and getattr(self, "add_v_proj", None) is not None
        ):
            concatenated_weights = torch.cat(
                [self.add_q_proj.weight.data, self.add_k_proj.weight.data, self.add_v_proj.weight.data]
            )
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_added_qkv = nn.Linear(
                in_features, out_features, bias=self.added_proj_bias, device=device, dtype=dtype
            )
            self.to_added_qkv.weight.copy_(concatenated_weights)
            if self.added_proj_bias:
                concatenated_bias = torch.cat(
                    [self.add_q_proj.bias.data, self.add_k_proj.bias.data, self.add_v_proj.bias.data]
                )
                self.to_added_qkv.bias.copy_(concatenated_bias)

        self.fused_projections = fuse


class QwenDoubleStreamAttnProcessor2_0WithStyleControl(nn.Module):
    """
    为 Qwen-Image-Edit 的双流架构设计的注意力处理器，增加了对风格控制图像的支持。
    该处理器将风格图像潜变量与主要内容（噪声+content_image_latents）分离，
    先执行标准的文本-主要内容联合注意力，再将风格信息注入到噪声部分。
    """
    _attention_backend = None
    _parallel_config = None
    def __init__(self, style_context_dim: int, style_hidden_dim: int):
        super().__init__()
        """
        Args:
            style_context_dim (`int`): 风格图像潜变量的维度 (e.g., 16 * 4 for Qwen's VAE latent channels * 4 from packing)。
            style_hidden_dim (`int`): 与 Qwen Transformer 内部计算相关的维度 (e.g., transformer.config.hidden_size or num_attention_heads * attention_head_dim)。
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0WithStyleControl requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )
        self.style_context_dim = style_context_dim
        self.style_hidden_dim = style_hidden_dim
        #print(f"Context: style_context_dim={style_context_dim}, style_hidden_dim={style_hidden_dim}")

        # 初始化用于风格控制的投影层
        # 输入维度是 style_image_latents 的最后一个维度
        self.style_k_proj = nn.Linear(style_context_dim, style_hidden_dim, bias=True)
        self.style_v_proj = nn.Linear(style_context_dim, style_hidden_dim, bias=True)
        self.style_scale = nn.Parameter(torch.tensor(100.0))
        # 初始化为零，符合 IP-Adapter 的做法
        # 用小的随机值初始化，这样训练初期就能看到风格控制的效果
        nn.init.normal_(self.style_k_proj.weight, std=0.01)
        nn.init.zeros_(self.style_k_proj.bias)
        nn.init.normal_(self.style_v_proj.weight, std=0.01) 
        nn.init.zeros_(self.style_v_proj.bias)
        #训练的时候要注释掉
        self.style_k_proj.to(dtype=torch.bfloat16,device="cuda")
        self.style_v_proj.to(dtype=torch.bfloat16,device="cuda")
        self.style_scale.to(dtype=torch.bfloat16,device="cuda")
        #print("Initialized QwenDoubleStreamAttnProcessor2_0WithStyleControl")
        

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # 图像流 (噪声 + content_image_latents + style_image_latents)
        encoder_hidden_states: torch.FloatTensor = None,  # 文本流
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs, # 通过 kwargs 捕获所有 attention_kwargs
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0WithStyleControl requires encoder_hidden_states (text stream)")

        # 从 kwargs 中提取风格相关信息
        style_image_latents = kwargs.get("style_image_latents", None) # [B, L_style, style_context_dim]
        style_start_idx = kwargs.get("style_start_idx", None) # style_image_latents 在 hidden_states 中的起始索引
        style_end_idx = kwargs.get("style_end_idx", None)   # style_image_latents 在 hidden_states 中的结束索引 (不包含)
        noise_patches_length = kwargs.get("noise_patches_length", None) # 噪声部分的 patch 数量
        content_patches_length = kwargs.get("content_patches_length", None) # 内容图像部分的 patch 数量
        #style_scale = kwargs.get("style_scale", 1.0) # 控制风格强度的缩放因子
        #print(f"style_start_idx: {style_start_idx}, style_end_idx: {style_end_idx}, noise_patches_length: {noise_patches_length}, content_patches_length: {content_patches_length}")

        seq_txt = encoder_hidden_states.shape[1]
        
        if isinstance(noise_patches_length, torch.Tensor):
            # 假设目标图都一样大
            noise_patches_length = noise_patches_length.flatten()[0].item()
        noise_patches_length = int(noise_patches_length) # 确保是 int
        
        if isinstance(style_start_idx, torch.Tensor):
            style_start_idx = style_start_idx.flatten()[0].item()
        style_start_idx = int(style_start_idx)

        if isinstance(style_end_idx, torch.Tensor):
            style_end_idx = style_end_idx.flatten()[0].item()

        # # --- 1. 执行标准的双流联合注意力，但只针对噪声+内容部分，排除风格 ---
        # # 首先分离噪声和内容部分（不包含风格）
        # if style_start_idx is not None and style_end_idx is not None:
        #     # 提取噪声+内容部分，排除风格部分
        #     noise_content_hidden_states = torch.cat([
        #         hidden_states[:, :style_start_idx, :],  # 噪声+内容部分
        #         hidden_states[:, style_end_idx:, :]     # 如果风格后面还有其他部分（通常没有）
        #     ], dim=1) if style_end_idx < hidden_states.shape[1] else hidden_states[:, :style_start_idx, :]
        # else:
        #     # 如果没有风格索引信息，假设风格在最后
        #     if noise_patches_length is not None and content_patches_length is not None:
        #         total_noise_content_len = noise_patches_length + content_patches_length
        #         noise_content_hidden_states = hidden_states[:, :total_noise_content_len, :]
        #     else:
        #         # 如果没有明确的长度信息，需要根据其他方式确定噪声+内容的长度
        #         raise ValueError("Either style indices or noise/content lengths must be provided")

        # 计算噪声+内容部分的 Q, K, V
        img_query_nc = attn.to_q(hidden_states) # [B, L_noise+L_content+L_style, D_qk]
        img_key_nc = attn.to_k(hidden_states)   # [B, L_noise+L_content+L_style, D_qk]
        img_value_nc = attn.to_v(hidden_states) # [B, L_noise+L_content+L_style, D_v]

        # 计算文本流的 Q, K, V
        txt_query = attn.add_q_proj(encoder_hidden_states) # [B, L_txt, D_qk]
        txt_key = attn.add_k_proj(encoder_hidden_states)   # [B, L_txt, D_qk]
        txt_value = attn.add_v_proj(encoder_hidden_states) # [B, L_txt, D_v]

        # 重塑以适应多头注意力
        img_query_nc = img_query_nc.unflatten(-1, (attn.heads, -1)) # [B, H, L_noise+L_content, D]
        img_key_nc = img_key_nc.unflatten(-1, (attn.heads, -1))  # [B, H, L_noise+L_content, D]
        img_value_nc = img_value_nc.unflatten(-1, (attn.heads, -1)) # [B, H, L_noise+L_content, D]
        txt_query = txt_query.unflatten(-1, (attn.heads, -1)) # [B, H, L_txt, D]
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))   # [B, H, L_txt, D]
        txt_value = txt_value.unflatten(-1, (attn.heads, -1)) # [B, H, L_txt, D]

        # 应用 QK 归一化
        if attn.norm_q is not None:
            img_query_nc = attn.norm_q(img_query_nc)
        if attn.norm_k is not None:
            img_key_nc = attn.norm_k(img_key_nc)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # 应用 RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            
            # 1. 确定切分点
            # style_start_idx: Style Image 开始的索引 (也是 Content 结束的地方)
            # noise_patches_length: Noise 结束的索引 (也是 Content 开始的地方)
            L_noise = int(noise_patches_length)
            idx_style_start = int(style_start_idx)
            
            # 2. 准备频率 (Frequencies)
            # [Set A] Noise 的坐标系 (0,0 -> H,W)。这是我们的"基准坐标"。
            freqs_noise = img_freqs[:L_noise] 
            
            # [Set B] Style 原本的坐标系。
            # Style 在序列的最后，它天然拥有偏移的坐标 (比如 t=1, 或者 h,w 很大)。
            # 我们直接用它原本的频率，利用这种天然的"平移"来防止结构泄露。
            # 注意：我们需要截取 img_freqs 中对应 Style 那一段的长度
            style_len = hidden_states.shape[1] - idx_style_start
            # 从 img_freqs 中取出对应 Style 的那一段频率
            freqs_style_original = img_freqs[idx_style_start : idx_style_start + style_len]

            # --- 3. 处理 Query (Q) ---
            # 我们主要关心 Noise 的 Q 带有位置信息，它是"手电筒"
            # 切片维度: [Batch, SeqLen, Heads, Dim] -> 切第1维
            q_noise   = img_query_nc[:, :L_noise, :, :]
            q_content = img_query_nc[:, L_noise:idx_style_start, :, :]
            q_style   = img_query_nc[:, idx_style_start:, :, :]
            
            # 给 Noise Q 加上基准坐标 (Set A)
            q_noise_roped = apply_rotary_emb_qwen(q_noise, freqs_noise, use_real=False)
            
            # Content 和 Style 的 Q 在生成过程中通常不重要 (因为我们取的是 Noise 的输出)
            # 为了不报错，可以不加 RoPE，或者加上对应的。这里选择保持原样(不加)以节省计算。
            q_content_roped = q_content 
            q_style_roped = q_style
            
            # 拼回 Query
            img_query_nc = torch.cat([q_noise_roped, q_content_roped, q_style_roped], dim=1)
            
            # --- 4. 处理 Key (K) - 这是魔法发生的地方 ---
            # 切片维度: [Batch, SeqLen, Heads, Dim] -> 切第1维
            k_noise   = img_key_nc[:, :L_noise, :, :]
            k_content = img_key_nc[:, L_noise:idx_style_start, :, :]
            k_style   = img_key_nc[:, idx_style_start:, :, :]
            
            # [关键点 1] Noise K 使用基准坐标 (Set A)
            # 自我认知正确
            k_noise_roped = apply_rotary_emb_qwen(k_noise, freqs_noise, use_real=False)
            
            # [关键点 2] Content K 强行使用 Noise 的坐标 (Set A) !!!
            # 欺骗模型：Content 的像素就在 Noise 的同一个位置。
            # 这会产生极强的对角线 Attention，锁死结构 (knight 的形状)。
            # 前提：Content 和 Noise 的分辨率必须一致 (通常 pipeline 处理后是一致的)
            k_content_roped = apply_rotary_emb_qwen(k_content, freqs_noise, use_real=False)
            
            # [关键点 3] Style K 使用它原本的坐标 (Set B)
            # 它的坐标天然就是平移过的 (比如 t 不同，或者在画布的其他位置)。
            # 这样 Q_noise 去查 K_style 时，位置对不上，无法复制形状，只能学习纹理。
            # 同时因为加了 RoPE，向量空间旋转是合法的，风格特征可以被计算。
            k_style_roped = apply_rotary_emb_qwen(k_style, freqs_style_original, use_real=False)
            
            # 拼回 Key
            img_key_nc = torch.cat([k_noise_roped, k_content_roped, k_style_roped], dim=1)
            
            # --- 5. 处理 Text RoPE (保持不变) ---
            if txt_freqs is not None:
                txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
        # 拼接进行联合注意力（文本 + 噪声+内容，不包含风格）
        joint_query = torch.cat([txt_query, img_query_nc], dim=1) # [B, H, L_txt+L_noise+L_content, D]
        joint_key = torch.cat([txt_key, img_key_nc], dim=1)   # [B, H, L_txt+L_noise+L_content, D]
        joint_value = torch.cat([txt_value, img_value_nc], dim=1) # [B, H, L_txt+L_noise+L_content, D]

        # 计算联合注意力
        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        ) # [B, H, L_txt+L_noise+L_content, D]

        # 重塑回原始格式
        joint_hidden_states = joint_hidden_states.flatten(2, 3) # [B, L_txt+L_noise+L_content, H*D]
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # 分离注意力输出
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # 文本部分 [B, L_txt, H*D]
        img_attn_output_nc = joint_hidden_states[:, seq_txt:, :]  # 图像部分 (噪声 + content) [B, L_noise+L_content+L_style, H*D]

        # 应用输出投影 (这部分是标准流程)
        img_attn_output_nc = attn.to_out[0](img_attn_output_nc)
        if len(attn.to_out) > 1:
            img_attn_output_nc = attn.to_out[1](img_attn_output_nc)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        # --- 2. 风格控制：如果提供了风格图像潜变量，则进行额外的风格注意力计算 ---
        img_attn_output_full = img_attn_output_nc  # 初始化为噪声+内容的输出
        if style_image_latents is not None and noise_patches_length is not None:
            # 从噪声+内容的注意力输出中提取噪声部分的 query（用于风格调制）
            # 注意：这里需要从原始的噪声+内容 hidden_states 计算噪声部分的 query
            #print(f"11111noise_content_hidden_states shape: {noise_patches_length}")
            #print(f"noise_patches_length: {noise_patches_length}")
            noise_hidden_states = hidden_states[:,:noise_patches_length, :] # [B, L_noise, D_hidden]
            #print(f"noise_hidden_states shape: {noise_hidden_states.shape}")
            img_query_noise = attn.to_q(noise_hidden_states).unflatten(-1, (attn.heads, -1)) # [B, H, L_noise, D]
            
            # 应用 Q 归一化
            if attn.norm_q is not None:
                img_query_noise = attn.norm_q(img_query_noise)
            
            # # 应用 RoPE - 只取噪声部分的位置编码
            # if image_rotary_emb is not None and img_freqs is not None:
            #     #print(f"noise img_freqs shape: {noise_patches_length, img_freqs.shape}")
            #     img_freqs_noise = img_freqs[:noise_patches_length, :]  # 只取噪声部分的编码
            #     img_query_noise = apply_rotary_emb_qwen(img_query_noise, img_freqs_noise, use_real=False)

            # 将 style_image_latents 投影为 K 和 V
            style_key = self.style_k_proj(style_image_latents) # [B, L_style, style_hidden_dim]
            style_value = self.style_v_proj(style_image_latents) # [B, L_style, style_hidden_dim]

            # 重塑 K 和 V 以适应多头
            style_key = style_key.unflatten(-1, (attn.heads, -1))  # [B, H, L_style, D]
            style_value = style_value.unflatten(-1, (attn.heads, -1))# [B, H, L_style, D]

            # 使用噪声部分的 Query 和风格图像的 K, V 进行注意力
            style_attention = F.scaled_dot_product_attention(
                img_query_noise, style_key, style_value, # Query 是噪声部分
                attn_mask=None, # 通常 style 不需要 mask
                dropout_p=0.0,
                is_causal=False
            ) # [B, H, L_noise_patches, D]

            # 重塑回原始格式
            style_attention = style_attention.flatten(2, 3) # [B, L_noise_patches, H*D]
            style_attention = style_attention.to(img_query_noise.dtype)

            # 将风格信息加到噪声部分上
            #print(f"style_attention shape: {style_attention.shape}, img_attn_output_full before shape: {img_attn_output_full.shape}")
            #print(f"style_scale: {self.style_scale}")
            #print(f"stylescale{self.style_scale}")
            img_attn_output_full[:, :noise_patches_length, :] = img_attn_output_full[:, :noise_patches_length, :] + 0 * style_attention

        

        # 返回完整的图像和文本注意力输出
        # final_img_output 的形状与输入 hidden_states 相同：[B, L_noise + L_content + L_style, H*D]
        return img_attn_output_full, txt_attn_output