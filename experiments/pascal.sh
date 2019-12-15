#!/usr/bin/env bash

cd ..

# test on single image
python main.py --image_path figures/pascal/2007_000876.jpg \
--model_path checkpoints/best_model_pascal_ResFCN.pth \
--model_name ResFCN