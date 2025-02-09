#!/bin/bash

# preprocessing
python main.py \
    --data_root '/home/mai/fke/fkee/coco2014' \
    --base_config './config/okvqa.yaml' \
    --model_config './config/model_config_1gpu.yaml' 
echo "preprocessing done"
