#!/bin/bash

# HS-FPN Enhanced A3Net for IU X-Ray Dataset
# 使用P3层输出（平衡性能和计算量）

python main.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 16 \
    --epochs 100 \
    --save_dir results/iu_xray_hsfpn1 \
    --step_size 50 \
    --gamma 0.1 \
    --seed 9233 \
    --use_hsfpn True \
    --hsfpn_output_layer P3 \
    --visual_extractor resnet101 \
    --visual_extractor_pretrained True \
    --d_model 512 \
    --d_ff 512 \
    --d_vf 2048 \
    --num_heads 8 \
    --num_layers 3 \
    --dropout 0.1 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --sample_method beam_search \
    --beam_size 3 \
    --n_gpu 1 \
    --lr_ve 5e-5 \
    --lr_ed 7e-4 \
    --weight_decay 5e-5 \
    --early_stop 30

