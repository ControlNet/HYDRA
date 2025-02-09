#!/bin/bash

python -m hydra_vl4ai.executor \
    --base_config ./config/okvqa.yaml \
    --model_config ./config/model_config_1gpu.yaml
 