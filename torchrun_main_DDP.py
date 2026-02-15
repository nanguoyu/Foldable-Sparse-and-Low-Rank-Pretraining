import os
import time
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from safetensors.torch import load_file
from contextlib import nullcontext

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
import datasets
import datasets.distributed

import wandb
from tqdm import tqdm
from loguru import logger

from para_eff_pt.peft_pretraining import training_utils
from para_eff_pt.peft_pretraining.dataloader import PreprocessedIterableDataset, PreprocessedIterableDataset_noslice


from para_eff_pt.peft_pretraining.dataloader_v2 import PreprocessedIterableDataset_v2


from para_eff_pt.peft_pretraining.modeling_llama import LlamaForCausalLM


from para_eff_pt.pt_sltrain import *
from para_eff_pt.pt_low_rank.low_rank_model import *
from para_eff_pt.pt_vrc.vrc_model import LowRankVirtualResidualChannelLinear

from para_eff_pt.utils.train_utils import *
from para_eff_pt.utils.args import parse_args
 
 

transformers.logging.set_verbosity_error()
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.enable_flash_sdp(True)


num_gpus = torch.cuda.device_count()
print(f"🔥Number of GPUs available: {num_gpus}")


@torch.no_grad()
def evaluate_model(
    args,
    model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, data_dir
):
    _time = time.time()
    # if not args.hf_dataset:
    #     logger.info(f"Using local dataset for validation")
    #     data_files_val= {"validation": [f"{args.dataset_path}/c4-validation.{str(i).zfill(5)}-of-00008.json.gz" for i in range(0,8)]}
    #     val_data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_val, split="validation", streaming=True)
    # else:
    if args.fast_val_dataset_path is not None:
        logger.info(f"🏠Using full validation dataset from {args.fast_val_dataset_path}")
        val_data = datasets.load_from_disk(args.fast_val_dataset_path)
    else:
        logger.info(f"Using full validation dataset from {data_dir}")
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=False, cache_dir=data_dir
        )  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=global_rank, world_size=world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
        val_data_mapped, batch_size
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens_tensor = torch.tensor(0, device=device, dtype=torch.long)
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens_tensor.item() > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens_tensor += (batch["input_ids"] != pad_idx).sum() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    # Convert back to int for return
    evaluated_on_tokens = evaluated_on_tokens_tensor.item()
    return total_loss, evaluated_on_tokens

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )
    logger.info(f"Total number of available GPUs: {torch.cuda.device_count()}")
    logger.info(f"world_size: {world_size}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    if global_rank == 0:
        logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            logger.info(f"gradient_accumulation: {args.gradient_accumulation} world_size: {world_size} total_batch_size: {args.total_batch_size} batch_size: {args.batch_size}")
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    if global_rank == 0:
        wandb.init(project=args.wandb_project_name, group=args.wandb_group_name, name=args.model_name)

        logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
        logger.info("*" * 40)
        logger.info(f"Starting training with the arguments")
        for k, v in vars(args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

    # data
    # if not args.hf_dataset:
    #     logger.info(f"Using local dataset for training")
    #     data_files_train = {"train": [f"{args.dataset_path}/c4-train.{str(i).zfill(5)}-of-01024.json.gz" for i in range(0,1024)]}
    #     logger.info(f"loading dataset")
    #     data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_train, split="train", streaming=True)
    #     logger.info(f"loaded dataset")
    # else:
    if args.fast_dataset_path is not None:
        logger.info(f"🏠Using fast dataset from {args.fast_dataset_path}")
        data = datasets.load_from_disk(args.fast_dataset_path)
    else:
        logger.info(f"Using full dataset from {args.data_dir} and then sample")
        data = datasets.load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=False,
            cache_dir=f"{args.data_dir}",
        )


    seed_for_shuffle = 42
    

    if global_rank == 0:
        logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data,
            rank=global_rank,
            world_size=world_size,
        )
    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length,token=False,
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

 
    if args.continue_from is not None:
        dataset = PreprocessedIterableDataset_v2(
            data, tokenizer, batch_size=args.batch_size, max_length=args.max_length, start_tokenizing_idx = args.start_tokenizing_idx
        )
    else:
        if args.no_slice: # !!!
            logger.info(f"Using PreprocessedIterableDataset_noslice !!")
            dataset = PreprocessedIterableDataset_noslice(
                data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
            )
        else:
             dataset = PreprocessedIterableDataset(
                data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
            )           



    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=args.workers,pin_memory=True
    )
 

    # model
    model_config = AutoConfig.from_pretrained(args.model_config)
    model_config.attn_implementation = "flash_attention_2"
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ====== starting config ======= #
    target_modules_list = ["attn", "mlp", "attention"]
    args.target_modules = target_modules_list

    # build model
    if args.dtype in ["bf16", "bfloat16"]:
        logger.info("Using bfloat16")
        model = build_model(model.to(device=device, dtype=torch.bfloat16), args)
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = build_model(model.to(device=device), args)
        model = model.to(device=device)


    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if 'spam' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        spam_params = []
        target_modules_list = ["attn", "mlp","attention"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            spam_params.append(module.weight)
        id_spam_params = [id(p) for p in spam_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_spam_params]

        # then call spam_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': spam_params, 'density': args.density, 'update_proj_gap': args.update_gap}]

    if ("galore" in args.optimizer.lower()) or ("fira" in args.optimizer.lower()):
        
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            
            
            if not ((isinstance(module, nn.Linear)) or (module.__class__.__name__ == 'SpLoRaLinear') or (module.__class__.__name__=='Restart_LoRaLinear')  ):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"enable {args.optimizer.lower()} for weights in module: ", module_name)
            
            if module.__class__.__name__ == 'SpLoRaLinear' or module.__class__.__name__ == 'Restart_LoRaLinear' :
                galore_params.append(module.lora_A)  
                
            else:
                galore_params.append(module.weight)
            
            
            
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [
            p for p in model.parameters() if id(p) not in id_galore_params
        ]
        
        opt_rank = args.rank
        if args.opt_rank != -1:
            opt_rank = args.opt_rank
        
        
        # then call galore_adamw
        param_groups = [
            {"params": regular_params},
            {
                "params": galore_params,
                "rank": opt_rank,
                "update_proj_gap": args.update_proj_gap,
                "scale": args.galore_scale,
                "proj_type": args.proj_type,
            },
        ]
    elif "apollo" in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        lowrank_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            #if not (isinstance(module, nn.Linear)):
            if not ((isinstance(module, nn.Linear)) or (module.__class__.__name__ == 'SpLoRaLinear') or (module.__class__.__name__=='Restart_LoRaLinear')  ) :    
            #if not (isinstance(module, nn.Linear) or isinstance(module, QScaleLinear) or isinstance(module, QLinear)):  
                continue
            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            logger.info(f"Adding {module_name} to APOLLO parameters")
            
            if module.__class__.__name__ == 'SpLoRaLinear' or module.__class__.__name__ == 'Restart_LoRaLinear':
                lowrank_params.append(module.lora_A)
            else:
                lowrank_params.append(module.weight)

        id_lowrank_params = [id(p) for p in lowrank_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_lowrank_params]
        # then call low rank optimizer
        param_groups = [
            {"params": regular_params},
            {
                "params": lowrank_params,
                "rank": args.rank,
                "update_proj_gap": args.update_proj_gap,
                "scale": args.apollo_scale,
                "proj_type": args.proj_type,
                "proj": args.proj,
                "scale_type": args.scale_type,
            },
        ]


    # build optimizer

    if "galore" in args.optimizer.lower():
        optimizer = build_optimizer_galore(model, param_groups, id_galore_params, args)
    elif "fira" in args.optimizer.lower():
        optimizer = build_optimizer_fira(model, param_groups, id_galore_params, args)
    elif "apollo" in args.optimizer.lower():
        optimizer = build_optimizer_apollo(model, param_groups, id_lowrank_params, args)
    elif 'stable_spam' in args.optimizer.lower():
        optimizer = build_optimizer_stable_spam(model, param_groups,  args)  
    elif 'spam' in args.optimizer.lower():
        optimizer = build_optimizer_spam(model, param_groups,  args)  
    elif "golore" in args.optimizer.lower():
        optimizer = build_optimizer_golore(model, trainable_params, args)
        
    elif "loro" in args.optimizer.lower():
            # Identify low-rank parameters
            loro_params = []
            regular_params = []
            processed_params = set()  # Used to track already processed parameters
            
            for module_name, module in model.named_modules():
                if isinstance(module, LoRaFaLinear):
                    print("Enable LORO for weights in module: ", module_name)
                    # Directly use lora_A and lora_B modules instead of their weights
                    loro_params.append({
                        'A': module.lora_A,
                        'B': module.lora_B,
                        'name': module_name
                    })
                    # Add processed parameter IDs
                    processed_params.add(id(module.lora_A.weight))
                    processed_params.add(id(module.lora_B.weight))
                    if module.bias is not None:
                        processed_params.add(id(module.bias))
            
            # Collect other parameters
            for name, param in model.named_parameters():
                if param.requires_grad and id(param) not in processed_params:
                    regular_params.append(param)
                    processed_params.add(id(param))
            
            # Print parameter statistics for debugging
            print(f"Found {len(loro_params)} LORO parameter pairs")
            print(f"Found {len(regular_params)} regular parameters")
            
            # Create optimizer group for LORO parameters
            loro_optimizer_params = []
            for param_dict in loro_params:
                loro_optimizer_params.extend([
                    param_dict['A'].weight,
                    param_dict['B'].weight
                ])
            
            param_groups = [
                {"params": regular_params},
                {
                    "params": loro_optimizer_params,
                    "update_k": args.cycle_length,
                    "is_loro": True
                }
            ]
            
            optimizer = build_optimizer_loro(model, param_groups, args)
            
     
        
    else:
        optimizer = build_optimizer(model, trainable_params, args)
       

    layer_wise_flag = True if "per_layer" in args.optimizer.lower() else False
    if layer_wise_flag:
        if not isinstance(optimizer, dict):
            raise ValueError("Layer-wise optimizer is not properly constructed.")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=args.cycle_length,
            recovery_steps=args.recovery_steps,

        )

    # Default: do not pre-skip
    pre_skip_micro = 0

    if args.continue_from is not None:

        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        safetensors_file = os.path.join(args.continue_from, "model.safetensors")

        # Create pytorch_model.bin from safetensors ONCE (rank 0), then barrier
        if not os.path.exists(checkpoint_path):
            if global_rank == 0:
                if not os.path.exists(safetensors_file):
                    raise FileNotFoundError(
                        f"Neither {checkpoint_path} nor {safetensors_file} exists in {args.continue_from}"
                    )
                state_dict = load_file(safetensors_file)
                tmp_path = checkpoint_path + ".tmp"
                torch.save(state_dict, tmp_path)
                os.replace(tmp_path, checkpoint_path)
                logger.info(
                    f"safetensors {safetensors_file} converted to pytorch bin {checkpoint_path} (rank 0)"
                )
            if dist.is_initialized():
                dist.barrier()

        # Try loading; if the .bin is corrupted (e.g., concurrent write), retry by regenerating from safetensors via rank 0
        def _load_state_dict_from_bin(path: str):
            try:
                return torch.load(path, map_location="cpu")
            except Exception as e:
                msg = str(e)
                if ("PytorchStreamReader" in msg) or ("failed finding central directory" in msg):
                    logger.warning(
                        f"Failed to read {path} ({e}). Attempting to regenerate from safetensors..."
                    )
                    if global_rank == 0:
                        if not os.path.exists(safetensors_file):
                            raise
                        state_dict = load_file(safetensors_file)
                        tmp_path = path + ".tmp"
                        torch.save(state_dict, tmp_path)
                        os.replace(tmp_path, path)
                        logger.info(f"Regenerated {path} from {safetensors_file}")
                    if dist.is_initialized():
                        dist.barrier()
                    return torch.load(path, map_location="cpu")
                raise

        state_dict_cpu = _load_state_dict_from_bin(checkpoint_path)

        if args.peft_model.lower() in ["sltrain", "fosl"]:
            model.wrapped_model.load_state_dict(state_dict_cpu, strict=True)
        else:
            model.load_state_dict(state_dict_cpu, strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")
        # Ensure all ranks have finished loading before proceeding
        if dist.is_initialized():
            dist.barrier()

        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu"
        )
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )
        else:
            logger.warning(
                f"Did not find training state in {args.continue_from}, global step will start from zero"
            )
        logger.info("*" * 40)

        # Compute whether we can safely pre-skip micro-steps in lockstep
        try:
            prev_cfg = optimizer_checkpoint.get("config", {})
            prev_ws = int(prev_cfg.get("world_size", world_size))
            prev_total_bs = int(prev_cfg.get("total_batch_size", args.total_batch_size))
            prev_bs = int(prev_cfg.get("batch_size", args.batch_size))
            prev_ga = prev_cfg.get("gradient_accumulation", None)
            if prev_ga is None:
                prev_ga = prev_total_bs // (prev_bs * prev_ws) if (prev_bs and prev_ws) else args.gradient_accumulation
            prev_ga = int(prev_ga)

            resume_can_skip = (prev_ws == world_size) and (prev_ga == args.gradient_accumulation)
            if resume_can_skip:
                pre_skip_micro = int(update_step) * int(args.gradient_accumulation)
                logger.info(
                    f"Lockstep pre-skip enabled: world_size/GA match (prev_ws={prev_ws}, prev_ga={prev_ga}); "
                    f"will pre-skip {pre_skip_micro} micro-steps."
                )
            else:
                pre_skip_micro = 0
                logger.info(
                    f"Lockstep pre-skip disabled due to topology mismatch or GA change: "
                    f"prev_ws={prev_ws}, curr_ws={world_size}, prev_ga={prev_ga}, curr_ga={args.gradient_accumulation}"
                )
        except Exception as e:
            pre_skip_micro = 0
            logger.warning(f"Failed to compute pre-skip micro-steps: {e}. Skipping pre-skip.")

    scheduler_start_step = update_step
    
    # Update tokens_seen_tensor after potential checkpoint restoration
    if args.continue_from is not None:
        logger.info(f"Updating tokens_seen_tensor with restored value: {tokens_seen}")
        tokens_seen_tensor = torch.tensor(tokens_seen, device=device, dtype=torch.long)
    else:
        tokens_seen_tensor = torch.tensor(tokens_seen, device=device, dtype=torch.long)

    logger.info("Compiling model with torch.compile")
    model = torch.compile(model)
    # Ensure all ranks have compiled before wrapping with DDP
    if dist.is_initialized():
        dist.barrier()

    # print params and trainable params
    logger.info(f"Running with {args.peft_model}\n")
    logger.info(f"\n{model}\n")
    logger.info(
        f"All params: \n{[n for n,p in model.named_parameters() if p.requires_grad]}\n"
    )
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Total non-low-rank and non-sparse parameters: "
        f"{sum(p.numel() for n,p in model.named_parameters() if 'lora_' not in n and 'sparse_' not in n) / 1_000_000:.2f}M"
    )

    if args.peft_model.lower() == "sltrain":
        logger.info(
            f"Total low-rank parameters: "
            f"{sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n) / 1_000_000:.2f}M"
        )
        logger.info(
            f"Total low-rank parameters (requires_grad): "
            f"{sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n and p.requires_grad) / 1_000_000:.2f}M"
        )
        logger.info(
            f"Total sparse parameters: "
            f"{sum(p.numel() for n, p in model.named_parameters() if 'sparse_' in n and p.requires_grad) / 1_000_000:.2f}M"
        )

    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "allenai/c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        pbar = tqdm(
            total=args.num_training_steps - update_step, desc="Update steps", ncols=80
        )

    if not args.single_gpu:
        # Optional sanity check: identical parameter counts across ranks
        try:
            numel_local = sum(p.numel() for p in model.parameters())
            numel_t = torch.tensor(numel_local, device=f"cuda:{local_rank}", dtype=torch.long)
            gathered = [torch.zeros_like(numel_t) for _ in range(world_size)]
            dist.all_gather(gathered, numel_t)
            base = int(gathered[0].item())
            if not all(int(x.item()) == base for x in gathered):
                raise RuntimeError(f"Parameter count mismatch across ranks: {[int(x.item()) for x in gathered]}")
        except Exception as _e:
            logger.warning(f"Param-count cross-rank check skipped/failed: {_e}")

        # Final sync before entering DDP to avoid param-shape verification races
        if dist.is_initialized():
            dist.barrier()

        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            # find_unused_parameters=False if "full" in args.model_name.lower() else True,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    # For global token-weighted training loss logging
    train_loss_num_local = torch.tensor(0.0, device=device, dtype=torch.float64)
    train_token_den_local = torch.tensor(0, device=device, dtype=torch.long)
    
    # Track tokens_seen as tensor to avoid frequent .item() calls
    # Note: tokens_seen_tensor will be properly initialized after checkpoint restoration

    # Prepare data iterator and perform optional lockstep pre-skip before training loop
    data_iter = iter(dataloader)
    if args.continue_from is not None:
        if pre_skip_micro > 0:
            if global_rank == 0:
                logger.info(f"Pre-skipping {pre_skip_micro} micro-steps to align data position with checkpoint")
            skipped = 0
            while skipped < pre_skip_micro:
                try:
                    next(data_iter)
                    skipped += 1
                    if skipped % 1000 == 0 and local_rank == 0:
                        print(skipped)
                except StopIteration:
                    logger.warning(
                        f"Data iterator exhausted during pre-skip at {skipped}/{pre_skip_micro}. Proceeding to training."
                    )
                    break
        # Common sync point for all ranks in resume path
        if dist.is_initialized():
            dist.barrier()

    # ##############################
    # TRAINING LOOP
    # ##############################

    grad_norm_prev = None

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory / 1024**3:.2f} GB\n")
    torch.cuda.reset_peak_memory_stats()


    # Log weights and optimizer memory once after the first optimizer step
    logged_weight_opt_mem = False
     
    for batch_idx, batch in enumerate(data_iter):
        if update_step == 0 and args.eval_at_begining :
            logger.info(f"Performing evaluation at step {update_step}")
            # Ensure eval mode to disable dropout, then restore train mode
            was_training = model.training
            model.eval()
            total_loss, evaluated_on_tokens = evaluate_model(
                args,
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
                args.data_dir,
            )
            if was_training:
                model.train()
            if global_rank == 0:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_perplexity": np.exp(total_loss),
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    # step=global_step,
                    step=update_step,
                )
            logger.info(
                f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
            )        


        # Some datasets may yield placeholder integers (e.g., PreprocessedIterableDataset_v2 before start_tokenizing_idx)
        # Skip such batches without advancing steps to keep GA boundaries intact.
        if not isinstance(batch, dict):
            continue

        global_step += 1
        local_step += 1
        

        if update_step > args.num_training_steps:
            logger.info(
                f"Reached max number of update steps (f{args.num_training_steps}). Stopping training."
            )
            print(f"Rank {global_rank} stopping training.")
            break
        
        if args.peft_model.lower() == 'golore':
            reset_relora = update_step % args.update_proj_gap == 0
            unwrapped_model = model.module if hasattr(model, 'module') else model
            unwrapped_model._config.forward_type = reset_relora

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        # Track local tokens (do not multiply by world_size)
        local_valid_tokens = (batch["input_ids"] != pad_idx).sum()
        tokens_seen_tensor += local_valid_tokens

        # Use DDP no_sync on non-final micro-steps to reduce comm overhead
        sync_ctx = nullcontext()
        if (not args.single_gpu) and hasattr(model, "no_sync") and (global_step % args.gradient_accumulation != 0):
            sync_ctx = model.no_sync()
        with sync_ctx:
            loss = model(**batch, labels=labels).loss
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()

        # Accumulate token-weighted numerators/denominators for global loss logging
        train_loss_num_local += loss.detach() * local_valid_tokens
        train_token_den_local += local_valid_tokens

        if global_step % args.gradient_accumulation != 0:
            continue
        grad_norm = None
        if args.grad_clipping != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
        else:
            with torch.no_grad():
                norms = [p.grad.detach().norm() for p in trainable_params if p.grad is not None]
                grad_norm = torch.norm(torch.stack(norms)) if norms else torch.tensor(0., device=device)
        
           
        if args.rand_ratio > 0.0 and args.peft_model.lower() == 'golore' and reset_relora:
            _lora_reset_time = time.time()
            # logger.info(f"{args.resume_from=}, {local_step=}, {args.relora=}, thresh: {local_step // args.gradient_accumulation}")
            logger.info(f"Performing lora reset at update step {update_step}. Current lr is {optimizer.param_groups[0]['lr']}")

            use_rand = update_step / args.num_training_steps >= args.rand_ratio
            
            underlying_model = model.module if hasattr(model, 'module') else model
            
            underlying_model.merge_and_reinit(optimizer,rand=use_rand)
            
            
        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            optimizer.step()    
            scheduler.step()
            optimizer.zero_grad()
            
        
        #print(gradnorms)

        update_step += 1
        update_time = time.time() - update_time
        
        # Log per-rank weights and optimizer state memory once (after first step)
        if (not logged_weight_opt_mem) and (update_step == 1):
            # Parameters (weights) memory
            param_bytes = 0
            for p in model.parameters():
                param_bytes += p.numel() * p.element_size()
            # Optimizer states memory (handle both single optimizer and per-layer dict)
            opt_bytes = 0
            if not layer_wise_flag:
                for st in optimizer.state.values():
                    for v in st.values():
                        if torch.is_tensor(v):
                            opt_bytes += v.numel() * v.element_size()
            else:
                for opt_i in optimizer.values():
                    for st in opt_i.state.values():
                        for v in st.values():
                            if torch.is_tensor(v):
                                opt_bytes += v.numel() * v.element_size()
            if global_rank == 0:
                logger.info(
                    f"Memory (per rank) after first step -> weights: {param_bytes / (1024**3):.2f} GB, "
                    f"optimizer_state: {opt_bytes / (1024**3):.2f} GB"
                )
                try:
                    wandb.log(
                        {
                            "weights_memory_GB": param_bytes / (1024**3),
                            "optimizer_state_memory_GB": opt_bytes / (1024**3),
                        },
                        step=update_step,
                    )
                except Exception:
                    pass
            logged_weight_opt_mem = True

        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
            and global_rank == 0
        ):
            if args.keep_only_last_model:
                current_model_directory = f"{args.save_dir}/model_last"
            else:
                current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(
                f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(
                current_model_directory, max_shard_size="100GB"
            )

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            was_training = model.training
            model.eval()
            total_loss, evaluated_on_tokens = evaluate_model(
                args,
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
                args.data_dir,
            )
            if was_training:
                model.train()
            if global_rank == 0:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_perplexity": np.exp(total_loss),
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    # step=global_step,
                    step=update_step,
                )

            logger.info(
                f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
            )


            


        
            
        # if update_step % args.cycle_length == 0 and ((args.peft_model.lower() == 'sltrain') or (args.peft_model.lower() == 'vrc')):
        if (
            # update_step >= args.warmup_steps                      
            # and 
            update_step % args.cycle_length == 0
            and (
                args.peft_model.lower() == "sltrain"
                # or args.peft_model.lower() == "vrc"
                # or args.peft_model.lower() == "mlow-rank"
                # or args.peft_model.lower() == "fosl"

            )
        ):
            logger.info(f"\nReinitialize B,A at update step {update_step}")
            
            ## There is a bug with the merge & reinitialize operation on FSDP, and the solution is currently unknown. Use DDP for now.
            '''
            RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet. 
            Caffe2 uses a lazy allocation, so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.
            '''
                        
            underlying_model = model.module if hasattr(model, 'module') else model
            for name, module in underlying_model.named_modules():   
                if isinstance(module, SpLoRaLinear) or isinstance(module, LowRankVirtualResidualChannelLinear):
                    module.merge_and_reinit()
                    
            
            new_params = [p for p in model.parameters() if p.requires_grad]
            
            args.lr = args.lr 
            num_training_steps = args.num_training_steps  # Use total steps
            
            
            if "fira" in args.optimizer.lower():
                optimizer = build_optimizer_fira(model, param_groups, id_galore_params, args)
            elif "apollo" in args.optimizer.lower():
                optimizer = build_optimizer_apollo(model, param_groups, id_lowrank_params, args)
            else:
                optimizer = build_optimizer(model, new_params, args)
            
            # Add initial_lr
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = args.lr  

                        
            scheduler = training_utils.get_scheculer(
                    optimizer=optimizer,
                    scheduler_type="cosine_quick_recovery",
                    num_training_steps=num_training_steps,  # Use total steps
                    warmup_steps=args.warmup_steps,  # No impact
                    min_lr_ratio=args.min_lr_ratio,
                    cycle_length=args.cycle_length,  # Restart interval
                    last_epoch=update_step,
                    recovery_steps=args.recovery_steps,
                )


        
        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer.values())[0].param_groups[0]["lr"]
        
        # Update tokens_seen only at update boundary (avoid frequent .item() calls)
        tokens_seen = tokens_seen_tensor.item()
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # Compute and log a global token-weighted loss at update boundary
        if global_step % args.gradient_accumulation == 0:
            # Use existing tensors directly instead of recreating
            if world_size > 1:
                dist.all_reduce(train_loss_num_local, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_token_den_local, op=dist.ReduceOp.SUM)
            loss_global = (train_loss_num_local / train_token_den_local.clamp(min=1)).item()
            
            # Optionally compute true global tokens seen in this window
            tokens_window_t = torch.tensor(tokens_in_update, device=device, dtype=torch.long)
            if world_size > 1:
                dist.all_reduce(tokens_window_t, op=dist.ReduceOp.SUM)
            throughput_tokens_global = float(tokens_window_t.item()) / update_time

            if global_rank == 0:
                wandb.log(
                    {
                        "loss": loss_global,
                        "lr": lr,
                        "update_step": update_step,
                        "tokens_seen": tokens_seen,
                        "throughput_tokens": throughput_tokens_global,
                        "throughput_examples": args.total_batch_size / update_time,
                        "throughput_batches": batches_in_update / update_time,
                        "gradnorm": grad_norm,
                        "max_memory": max_memory,
                    },
                    step=update_step,
                )
            # reset accumulators
            train_loss_num_local.zero_()
            train_token_den_local.zero_()

        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        args,
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
        args.data_dir,
    )

    if global_rank == 0:
        wandb.log(
            {
                "final_eval_loss": total_loss,
                "final_eval_perplexity": np.exp(total_loss),
                "final_eval_tokens": evaluated_on_tokens,
            },
            # step=global_step,
            step=update_step,
        )
        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
        )


    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
