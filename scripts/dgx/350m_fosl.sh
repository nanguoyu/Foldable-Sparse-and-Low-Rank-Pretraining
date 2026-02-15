#!/bin/bash

# --- set default values -----------
cuda_idx=0,3,4,7
cuda_num=4
batch_size=64
global_batch_size=512
rank=256
lora_alpha=16
lr=0.003
beta1=0.9
beta2=0.99
adamw_epsilon=1e-7
cycle_length=650000
folding_ratio=0.9
lr_act=True
sparse_trainable_scaling=False
mix_trainable=True
mix_per_channel=False
mix_init=0.7
init_lost=False

# --- parse CLI overrides -----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sparse_trainable_scaling=*)
            sparse_trainable_scaling="${1#*=}"
            shift
            ;;
        --sparse_trainable_scaling)
            sparse_trainable_scaling="$2"
            shift 2
            ;;
        --mix_trainable=*)
            mix_trainable="${1#*=}"
            shift
            ;;
        --mix_trainable)
            mix_trainable="$2"
            shift 2
            ;;
        --mix_per_channel=*)
            mix_per_channel="${1#*=}"
            shift
            ;;
        --mix_per_channel)
            mix_per_channel="$2"
            shift 2
            ;;
        --mix_init=*)
            mix_init="${1#*=}"
            shift
            ;;
        --mix_init)
            mix_init="$2"
            shift 2
            ;;
        --init_lost=*)
            init_lost="${1#*=}"
            shift
            ;;
        --init_lost)
            init_lost="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--sparse_trainable_scaling True|False] [--mix_trainable True|False] [--mix_per_channel True|False] [--mix_init 0.7] [--init_lost True|False]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Efficient pre-training with $cuda_num |GPU(s): $cuda_idx |rank: $rank | batch_size: $batch_size | global_batch_size: $global_batch_size | lr: $lr"


CUDA_VISIBLE_DEVICES=$cuda_idx torchrun --standalone --nproc_per_node $cuda_num torchrun_main_DDP.py \
    --model_name "fosl rank:$rank ngpu:$cuda_num lr:$lr folding_ratio:$folding_ratio lr_act:$lr_act sparse_trainable_scaling:$sparse_trainable_scaling mix_trainable:$mix_trainable mix_per_channel:$mix_per_channel mix_init:$mix_init init_lost:$init_lost b2:$beta2 esp:$adamw_epsilon" \
    --wandb_project_name "Efficient_Pretraining" \
    --wandb_group_name "llama_350m" \
    --model_config configs/llama_350m.json \
    --data_dir ~/data \
    --lr $lr \
    --b1 $beta1 \
    --b2 $beta2 \
    --adamw_epsilon $adamw_epsilon \
    --grad_clipping 0.5 \
    --peft_model fosl \
    --folding_ratio $folding_ratio \
    --lr_act $lr_act \
    --sparse_trainable_scaling $sparse_trainable_scaling \
    --mix_trainable $mix_trainable \
    --mix_per_channel $mix_per_channel \
    --mix_init $mix_init \
    --optimizer adamw \
    --rank $rank \
    --train_scaling \
    --lora_alpha $lora_alpha \
    --batch_size $batch_size \
    --total_batch_size $global_batch_size \
    --cycle_length $cycle_length \
    --num_training_steps 65000 \
    --warmup_steps 6500 \
    --weight_decay 0 \
    --dtype "bfloat16" \
    --eval_every 1000