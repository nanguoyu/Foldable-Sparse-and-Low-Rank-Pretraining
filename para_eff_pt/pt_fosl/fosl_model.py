import os
import math
import json

from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.activations import ACT2FN

@dataclass
class foslConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    trainable_scaling: bool = False
    folding_ratio: float = 0.9
    lr_act: bool = True
    lr_act_type: str = "silu"
    sparse_trainable_scaling: bool = False
    mix_trainable: bool = False
    mix_per_channel: bool = False
    mix_init: float = 0.7
    init_lost: bool = False
    lost_svd_rank: int = 256


class foslModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.0,
        trainable_scaling=False,
        folding_ratio=0.9,
        lr_act=True,
        lr_act_type="silu",
        sparse_trainable_scaling=False,
        mix_trainable=False,
        mix_per_channel=False,
        mix_init=0.7,
        init_lost=False,
        lost_svd_rank=256,
        fine_tuning=False,
    ):
        if r <= 0:
            raise ValueError("r must be positive (r >= 1).")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
        self.trainable_scaling = trainable_scaling
        self.target_modules = target_modules
        self.parameterized_modules = []
        assert 0 < folding_ratio <= 1, "folding_ratio must be between 0 and 1"
        # persist options for submodule construction
        self.folding_ratio = folding_ratio
        self.lr_act = lr_act
        self.lr_act_type = lr_act_type
        self.sparse_trainable_scaling = sparse_trainable_scaling
        self.mix_trainable = mix_trainable
        self.mix_per_channel = mix_per_channel
        self.mix_init = mix_init
        self.init_lost = init_lost
        self.lost_svd_rank = lost_svd_rank
        self.fine_tuning = fine_tuning
        self._config = foslConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            trainable_scaling=trainable_scaling,
            folding_ratio=folding_ratio,
            lr_act=lr_act,
            lr_act_type=lr_act_type,
            sparse_trainable_scaling=sparse_trainable_scaling,
            mix_trainable=mix_trainable,
            mix_per_channel=mix_per_channel,
            mix_init=mix_init,
            init_lost=init_lost,
            lost_svd_rank=lost_svd_rank,
        )

        # expose common HF attributes directly on wrapper for compatibility
        # (keep a direct reference so hasattr checks succeed without __getattr__)
        try:
            self.config = model.config  # type: ignore[attr-defined]
        except Exception:
            pass

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"Reparameterized module: {module_name}")
            new_module = foslLinear(
                module.in_features,
                module.out_features,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                trainable_scaling=self.trainable_scaling,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
                folding_ratio=self.folding_ratio,
                lr_act=self.lr_act,
                lr_act_type=self.lr_act_type,
                sparse_trainable_scaling=self.sparse_trainable_scaling,
                mix_trainable=self.mix_trainable,
                mix_per_channel=self.mix_per_channel,
                mix_init=self.mix_init,
                init_lost=self.init_lost,
                lost_svd_rank=self.lost_svd_rank,
                orig_weight=module.weight.data.clone() if self.fine_tuning else None,
                orig_bias=(module.bias.data.clone() if (self.fine_tuning and module.bias is not None) else None),
                finetune_mode=self.fine_tuning,
            )

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def __getattr__(self, name):
        """
        Delegate missing attributes to the wrapped HF model so code expecting
        `model.config`, `model.base_model`, etc. keeps working.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped_model = super().__getattribute__("wrapped_model")
            return getattr(wrapped_model, name)

    def forward(self, *args, **kwargs):
        """Call the wrapped model's forward without rebinding `self`."""
        return self.wrapped_model(*args, **kwargs)

    def set_finetune_mode(self, enabled: bool = True):
        """
        Toggle fine-tuning forward mode for all foslLinear modules.
        If enabled but a layer has no stored weight_original, that layer will silently skip
        the original path contribution.
        """
        for _, module in self.wrapped_model.named_modules():
            if isinstance(module, foslLinear):
                module.finetune_mode = enabled

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def save_pretrained(self, path, max_shard_size="100GB"):
        # TODO
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "fosl_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        # TODO
        with open(os.path.join(path, "fosl_config.json"), "r") as f:
            lorafa_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in lorafa_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {lorafa_config['keep_original']}")
            lorafa_config["lora_only"] = not lorafa_config.pop("keep_original")
            lorafa_config["keep_original_weights"] = not lorafa_config["lora_only"]

        if "trainable_scaling" not in lorafa_config:
            lorafa_config["trainable_scaling"] = False

        model = cls(base_model, **lorafa_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


class foslLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
        folding_ratio: float = 0.9,
        lr_act=True,
        lr_act_type="silu",
        sparse_trainable_scaling=False,
        mix_trainable=False,
        mix_per_channel=False,
        mix_init=0.7,
        init_lost=False,
        lost_svd_rank=256,
        orig_weight: Tensor | None = None,
        orig_bias: Tensor | None = None,
        finetune_mode: bool = False,
    ):
        """
        Reparameterized low rank linear layer
                    x W_a @ W_b * lora_alpha / r + x W_sparse + bias
        Note that W_sparse = [W_base, W_foldable], where W_base is the base part and W_foldable is the foldable part copied from W_base by a fixed mask
        Notice that scale = lora_alpha / r.
        Notice that this class cannot be wrapped to linear layer and thus cannot be used for fine-tune
        For fine-tune, please refer to ... TODO
        """
        super().__init__()
        # nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive (r >= 1).")
        assert 0 < folding_ratio <= 1, "folding_ratio must be between 0 and 1"
        if bias:
            self.bias = Parameter(
                torch.zeros(
                    out_features, device=device, dtype=dtype, requires_grad=True
                )
            )
            a = 1 / math.sqrt(out_features)
            nn.init.uniform_(self.bias, -a, a)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        # self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.folding_ratio = folding_ratio
        self.trainable_scaling = trainable_scaling
        self.device = device
        self.dtype = dtype
        self.sparse_trainable_scaling = sparse_trainable_scaling
        self.finetune_mode = finetune_mode
        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        # Initialize low-rank parameters (LoRA)
        self._initialize_lora(init_lost=init_lost, lost_svd_rank=lost_svd_rank, orig_weight=orig_weight)

        # ensure proper device and dtype
        if device is not None or dtype is not None:
            self.lora_A.to(device=device, dtype=dtype)
            self.lora_B.to(device=device, dtype=dtype)
        
        if trainable_scaling:
            self.scaling = nn.Parameter(
                torch.tensor([1.0], device=device, dtype=dtype), requires_grad=True
            )
        else:
            self.scaling = self.lora_alpha / self.r
        
        
        # Initialize the W_sparse weight, physically, there is no foldable weight, we just reuse a part of W_base's output as the W_foldable's output,
        self.foldable_out_features = int(out_features * folding_ratio)
        self.base_out_features = out_features - self.foldable_out_features
        self.W_base = nn.Linear(in_features, self.base_out_features, bias=False)
        # weights initialization separated
        base_idx = self._initialize_sparse_base(init_lost, lost_svd_rank, orig_weight)

        if device is not None or dtype is not None:
            self.W_base.to(device=device, dtype=dtype)

        mask, foldable_scaling, base_scaling = self._generate_fold_mask_and_scaling(
            init_lost=init_lost,
            lost_svd_rank=lost_svd_rank,
            orig_weight=orig_weight,
            base_idx=base_idx,
        )
        # Move to target device and match computation dtype to W_base
        if device is not None:
            mask = mask.to(device)
        comp_dtype = self.W_base.weight.dtype
        foldable_scaling = foldable_scaling.to(device=device, dtype=comp_dtype)
        base_scaling = base_scaling.to(device=device, dtype=comp_dtype)

        self.register_buffer("select_mask_from_base", mask)
        self.register_buffer("foldable_scaling_factors", foldable_scaling)
        self.register_buffer("base_scaling_factors", base_scaling)
        # Save original dense weights/bias as frozen buffers only for fine-tuning
        if finetune_mode and orig_weight is not None:
            self.register_buffer("weight_original", orig_weight.to(device=device, dtype=comp_dtype))
        if finetune_mode and orig_bias is not None:
            self.register_buffer("bias_original", orig_bias.to(device=device, dtype=comp_dtype))
        # In order to improve the representation capacity, we will train an additional scaling factor for all W_sparse output channels
        if self.sparse_trainable_scaling:
            self.W_sparse_scaling = nn.Parameter(
                torch.ones(self.out_features, device=device, dtype=dtype), requires_grad=True
            )
        
        # learnable mixing between LoRA and sparse paths
        self.mix_trainable = mix_trainable
        self.mix_per_channel = mix_per_channel
        if self.mix_trainable:
            init = torch.as_tensor(mix_init, device=device, dtype=dtype).clamp(1e-6, 1 - 1e-6)
            init_logit = torch.logit(init)
            if self.mix_per_channel:
                self.mix_logit = nn.Parameter(init_logit.expand(out_features).clone(), requires_grad=True)
            else:
                self.mix_logit = nn.Parameter(init_logit.clone(), requires_grad=True)
        else:
            if self.mix_per_channel:
                self.register_buffer("fixed_alpha", torch.full((out_features,), mix_init, device=device, dtype=dtype))
            else:
                self.register_buffer("fixed_alpha", torch.as_tensor(mix_init, device=device, dtype=dtype))
        # keep original weights for finetune forward if stored as buffers

    def _initialize_lora(self, init_lost: bool, lost_svd_rank: int, orig_weight: Tensor | None):
        """
        Initialize lora_A and lora_B. If init_lost is True and orig_weight is available,
        use LOST-style SVD init with fixed svd rank. Otherwise use default init.
        """
        if init_lost and orig_weight is not None:
            with torch.no_grad():
                W = orig_weight.detach().to(device=self.device, dtype=torch.float32)
                min_dim = min(W.shape[0], W.shape[1])
                svd_r = max(1, min(lost_svd_rank, min_dim))
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                s_root = torch.sqrt(S[:svd_r])
                A_in_r = Vh[:svd_r, :].T * s_root.unsqueeze(0)
                B_out_r = U[:, :svd_r] * s_root.unsqueeze(0)
                take_r = min(self.r, svd_r)
                A_fit = A_in_r[:, :take_r]
                B_fit = B_out_r[:, :take_r]
                if take_r < self.r:
                    pad_cols = self.r - take_r
                    A_fit = torch.cat([
                        A_fit,
                        torch.zeros(self.in_features, pad_cols, device=W.device, dtype=W.dtype),
                    ], dim=1)
                    B_fit = torch.cat([
                        B_fit,
                        torch.zeros(self.out_features, pad_cols, device=W.device, dtype=W.dtype),
                    ], dim=1)
                self.lora_A.weight.data.copy_(A_fit.T.to(self.dtype if self.dtype is not None else self.lora_A.weight.dtype))
                self.lora_B.weight.data.copy_(B_fit.to(self.dtype if self.dtype is not None else self.lora_B.weight.dtype))
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def _initialize_sparse_base(self, init_lost: bool, lost_svd_rank: int, orig_weight: Tensor | None):
        """
        Initialize W_base weights. Returns base_idx if LOST init is used, else None.
        """
        if init_lost and orig_weight is not None and self.base_out_features > 0:
            with torch.no_grad():
                W = orig_weight.detach().to(device=self.device, dtype=torch.float32)
                min_dim = min(W.shape[0], W.shape[1])
                svd_r = max(1, min(lost_svd_rank, min_dim))
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                if svd_r < min_dim:
                    U_res = U[:, svd_r:]
                    S_res = S[svd_r:]
                    Vh_res = Vh[svd_r:, :]
                    W_comp = (U_res * S_res.unsqueeze(0)) @ Vh_res
                else:
                    W_comp = torch.zeros_like(W)
                importance_out = torch.linalg.vector_norm(W_comp, dim=1)
                m = self.base_out_features
                base_idx = torch.topk(importance_out, k=m, largest=True, sorted=True).indices
                self.W_base.weight.data.copy_(W[base_idx, :].to(self.dtype if self.dtype is not None else self.W_base.weight.dtype))
                return base_idx
        else:
            nn.init.kaiming_uniform_(self.W_base.weight, a=math.sqrt(5))
            return None

    def _generate_fold_mask_and_scaling(
        self,
        *,
        init_lost: bool,
        lost_svd_rank: int,
        orig_weight: Tensor | None,
        base_idx: Tensor | None,
    ):
        """
        Generate the foldable selection mask and scaling tensors.
        If LOST init was used and base_idx is provided, build mapping using cosine similarity; otherwise use random balanced mapping.
        Returns (mask, foldable_scaling, base_scaling).
        """
        m = int(self.base_out_features)
        n = int(self.foldable_out_features)
        if m <= 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )

        if init_lost and orig_weight is not None and base_idx is not None:
            with torch.no_grad():
                W = orig_weight.detach().to(device=self.device, dtype=torch.float32)
                rest_mask = torch.ones(W.shape[0], dtype=torch.bool, device=W.device)
                rest_mask[base_idx] = False
                rest_rows = W[rest_mask, :]
                n_rows = rest_rows.shape[0]
                if n_rows > 0:
                    eps = 1e-8
                    base_norm = self.W_base.weight.data.to(torch.float32)
                    base_norm = base_norm / (base_norm.norm(dim=1, keepdim=True) + eps)
                    rest_norm = rest_rows.to(torch.float32)
                    rest_norm = rest_norm / (rest_norm.norm(dim=1, keepdim=True) + eps)
                    sim = rest_norm @ base_norm.T
                    assigned = torch.argmax(sim, dim=1)
                    # If we have fewer or more foldable channels than rest_rows, subsample or repeat
                    if n <= n_rows:
                        idx = torch.arange(n, device=assigned.device)
                        mask = assigned[idx]
                    else:
                        k = (n + n_rows - 1) // n_rows
                        rep = assigned.repeat(k)[:n]
                        mask = rep
                else:
                    mask = torch.empty(0, dtype=torch.long, device=W.device)
                usage_count = torch.zeros(m, dtype=torch.long, device=self.W_base.weight.device)
                if mask.numel() > 0:
                    usage_count.scatter_add_(0, mask, torch.ones_like(mask, dtype=torch.long))
                total_usage = usage_count + 1
                base_scaling = 1.0 / torch.sqrt(total_usage.float())
                foldable_scaling = base_scaling[mask] if mask.numel() > 0 else torch.empty(0, dtype=torch.float32, device=base_scaling.device)
                return mask, foldable_scaling, base_scaling

        # Default random mapping
        if n > 0:
            k = (n + m - 1) // m
            perms = [torch.randperm(m, dtype=torch.long) for _ in range(k)]
            mask = torch.cat(perms, dim=0)[:n]
            usage_count = torch.zeros(m, dtype=torch.long)
            usage_count.scatter_add_(0, mask, torch.ones_like(mask))
            total_usage = usage_count + 1
            base_scaling = 1.0 / torch.sqrt(total_usage.float())
            foldable_scaling = base_scaling[mask]
        else:
            mask = torch.empty(0, dtype=torch.long)
            foldable_scaling = torch.empty(0, dtype=torch.float32)
            base_scaling = torch.empty(m, dtype=torch.float32) if m > 0 else torch.empty(0, dtype=torch.float32)
        return mask, foldable_scaling, base_scaling

    def _generate_select_mask_from_base(self):
        """
        Build an index mask (1D LongTensor) selecting foldable_out_features columns
        from the base outputs (size base_out_features), and corresponding scaling factors
        to maintain proper variance when channels are reused.

        - If foldable_out_features <= base_out_features: sample without replacement.
        - If foldable_out_features > base_out_features: repeat a random permutation
          of base indices as many times as needed and then trim to the target size.

        Returns:
        - mask: index tensor for selecting channels
        - foldable_scaling: per-foldable-channel scaling factors (FloatTensor[n])
        - base_scaling: per-base-channel scaling factors (FloatTensor[m])
        """
        m = int(self.base_out_features)
        n = int(self.foldable_out_features)

        # No base channels to fold from; keep behavior consistent (empty index)
        if m <= 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )

        if n <= m:
            mask = torch.randperm(m, dtype=torch.long)[:n]
            # No reuse: foldable scaling = 1, base scaling = 1
            foldable_scaling = torch.ones(n, dtype=torch.float32)
            base_scaling = torch.ones(m, dtype=torch.float32)
            return mask, foldable_scaling, base_scaling

        # Old code
        # perm = torch.randperm(m, dtype=torch.long)
        # idx = torch.arange(n, dtype=torch.long) % m
        # mask = perm[idx]

        # New code
        # Use “multi-permutation concatenation”: generate k = ceil(n/m) independent permutations of [0, m), 
        # concatenate, and slice to length n. This keeps counts balanced (each base channel appears either floor(n/m) or ceil(n/m) times),
        #  avoids the periodic adjacency structure of repeating a single permutation, and works well with your existing 1/sqrt(total_usage) scaling.
        k = (n + m - 1) // m
        perms = [torch.randperm(m, dtype=torch.long) for _ in range(k)]
        mask = torch.cat(perms, dim=0)[:n]



        # Calculate usage counts: each base channel appears once in base_out + k times in foldable_out
        usage_count = torch.zeros(m, dtype=torch.long)
        usage_count.scatter_add_(0, mask, torch.ones_like(mask))
        # Total usage = 1 (from base) + foldable usage
        total_usage = usage_count + 1

        # Calculate scaling factors: 1/sqrt(total_usage) for variance normalization
        base_scaling = 1.0 / torch.sqrt(total_usage.float())
        foldable_scaling = base_scaling[mask]  # scaling for foldable channels

        return mask, foldable_scaling, base_scaling

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()
        return self.scaling

    def forward(self, x: Tensor):
        """
        Input x : [..., in_dim] and Output [..., out_dim]
        """
        # If in fine-tuning mode with stored original weights, compute original path first
        if getattr(self, "finetune_mode", False) and hasattr(self, "weight_original"):
            if hasattr(self, "bias_original"):
                original_out = F.linear(x, self.weight_original, self.bias_original)
            else:
                original_out = F.linear(x, self.weight_original, None)
        else:
            original_out = None

        # lora part
        if hasattr(self, "lr_act") and self.lr_act:
            lora_out = self.lora_B(self.lr_act(self.lora_A(x))) * self._post_lora_scale()
        else:
            lora_out = self.lora_B(self.lora_A(x)) * self._post_lora_scale()

        # sparse part with variance-corrected scaling
        base_out_raw = self.W_base(x)
        
        # Apply scaling to base output to account for channel reuse
        if hasattr(self, "base_scaling_factors"):
            base_out = base_out_raw * self.base_scaling_factors.unsqueeze(0)  # broadcast over batch dims
        else:
            base_out = base_out_raw
        
        # select from unscaled base output and apply foldable scaling
        foldable_out = base_out_raw.index_select(dim=-1, index=self.select_mask_from_base)
        if hasattr(self, "foldable_scaling_factors"):
            foldable_out = foldable_out * self.foldable_scaling_factors.unsqueeze(0)  # broadcast over batch dims
            
        sparse_out = torch.cat([base_out, foldable_out], dim=-1)
        if self.sparse_trainable_scaling:
            sparse_out = sparse_out * self.W_sparse_scaling
        # Fine-tuning: original + alpha*lora + (1-alpha)*sparse
        if original_out is not None:
            if hasattr(self, "mix_logit"):
                alpha = torch.sigmoid(self.mix_logit)
            else:
                alpha = self.fixed_alpha
            out = original_out + lora_out * alpha + sparse_out * (1 - alpha)
            return out.contiguous()

        # Pre-training: keep existing mixing behavior
        if hasattr(self, "mix_logit"):
            alpha = torch.sigmoid(self.mix_logit)
        else:
            alpha = self.fixed_alpha
        out = lora_out * alpha + sparse_out * (1 - alpha)

        if self.bias is not None:
            out += self.bias
        return out.contiguous()

   
    def extra_repr(self):
        return f"in_dim={self.in_features}, out_dim={self.out_features}, rank={self.r}, folding_ratio={self.folding_ratio}, lr_act={self.lr_act}, sparse_trainable_scaling={self.sparse_trainable_scaling}"
    
# To use fosl in fine-tuning

def apply_fosl_param(
    model,
    model_type,
    scope,
    rank,
    alpha,
    init,
    *,
    fine_tuning: bool = False,
    folding_ratio: float = 0.9,
    lr_act: bool = True,
    lr_act_type: str = "silu",
    sparse_trainable_scaling: bool = False,
    mix_trainable: bool = False,
    mix_per_channel: bool = False,
    mix_init: float = 0.7,
):
    """
    Wrap the given pretrained model with foslModel according to model_type and scope.
    - model_type: e.g., 'roberta-base', 'llama', etc. We use substring checks.
    - scope: selects which submodules to target (see mappings below).
    - rank, alpha, init: LoRA rank, LoRA alpha, and LOST init flag (bool).
    - fine_tuning: if True, enable forward = original + sparse + lora, with original frozen.
    Returns the wrapped model (foslModel instance).
    """

    if "roberta" in model_type:
        # Use precise path fragments to avoid pooler/classifier dense layers
        scope_to_targets = {
            "all": [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "intermediate.dense",
                "output.dense",
            ],
            "qkv": [
                "attention.self.query",
                "attention.self.value",
                "attention.self.key",
            ],
            "qv": [
                "attention.self.query",
                "attention.self.value",
            ],
        }
        if scope not in scope_to_targets:
            raise ValueError(f"Unsupported scope '{scope}' for roberta")
        target_modules = scope_to_targets[scope]
    elif "llama" in model_type or model_type == "llama":
        scope_to_targets = {
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "qv": ["q_proj", "v_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
        }
        if scope not in scope_to_targets:
            raise ValueError(f"Unsupported scope '{scope}' for llama")
        target_modules = scope_to_targets[scope]
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")

    wrapped = foslModel(
        model,
        target_modules=target_modules,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        trainable_scaling=False,
        folding_ratio=folding_ratio,
        lr_act=lr_act,
        lr_act_type=lr_act_type,
        sparse_trainable_scaling=sparse_trainable_scaling,
        mix_trainable=mix_trainable,
        mix_per_channel=mix_per_channel,
        mix_init=mix_init,
        init_lost=bool(init),
        lost_svd_rank=256,
        fine_tuning=fine_tuning,
    )
    return wrapped


def get_fosl_param(model, lr_scaler: float = 1.0):
    """
    Collect trainable adapter parameter groups for fosl.
    Returns three groups when present:
      - type=fosl_in:  all lora_A parameters
      - type=fosl_out: all lora_B parameters
      - type=fosl_sparse: all sparse adapter params (W_base.weight, W_sparse_scaling)
    Also freezes all other parameters in the provided model.
    """
    # Allow passing either foslModel or a plain model already wrapped with foslLinear modules
    search_root = model.wrapped_model if isinstance(model, foslModel) else model

    lora_in_params = []
    lora_out_params = []
    sparse_params = []

    for _, module in search_root.named_modules():
        if isinstance(module, foslLinear):
            # LoRA params
            lora_in_params.append(module.lora_A.weight)
            lora_out_params.append(module.lora_B.weight)
            # Sparse adapter params
            sparse_params.append(module.W_base.weight)
            if hasattr(module, "W_sparse_scaling"):
                sparse_params.append(module.W_sparse_scaling)
            # Optional trainable scaling for LoRA
            if hasattr(module, "scaling") and isinstance(module.scaling, nn.Parameter):
                lora_out_params.append(module.scaling)
            # Optional mixing parameter: train in both modes since alpha used in both
            if hasattr(module, "mix_logit"):
                sparse_params.append(module.mix_logit)

    # Freeze everything by default
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze adapter params
    for p in lora_in_params + lora_out_params + sparse_params:
        p.requires_grad_(True)

    param_groups = []
    if lora_in_params:
        param_groups.append({"type": "fosl_in", "params": lora_in_params, "lr_scaler": lr_scaler})
    if lora_out_params:
        param_groups.append({"type": "fosl_out", "params": lora_out_params, "lr_scaler": lr_scaler})
    if sparse_params:
        param_groups.append({"type": "fosl_sparse", "params": sparse_params, "lr_scaler": lr_scaler})

    return param_groups