#!/usr/bin/env bash

cd ..

# test on single image
python main.py --image_path figures/test.png \
--model_path checkpoints/best_model_trancos_ResFCN.pth \
--model_name ResFCN