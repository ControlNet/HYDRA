# <img src=media/HYDRA_icon_minimal.png width="30"> HYDRA:

**This is the code for the paper [HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning](https://arxiv.org/abs/2403.12884), , which has been accepted by ECCV 2024 [Project Page](https://hydra-vl4ai.github.io).**

<img src=media/Frame.png width="800"> 

## Release

- [2024/07/29] 🔥 **HYDRA** is open sourced in GitHub.

## TODOs
We realize that `gpt-3.5-turbo-0613` is deprecated, and `gpt-3.5` will be replaced by `gpt-4o-mini`. We will release another version of HYDRA.
>As of July 2024, `gpt-4o-mini` should be used in place of `gpt-3.5-turbo`, as it is cheaper, more capable, multimodal, and just as fast [Openai API Page](https://platform.openai.com/docs/models/gpt-3-5-turbo).

We also notice the embedding model is updated by OpenAI as shown in this [link](https://openai.com/index/new-embedding-models-and-api-updates/). Due to the uncertainty of the embedding model updates from OpenAI, we suggest you train a new version of the RL controller yourself and update the RL models.
- [x] GPT-4o-mini replacement.
- [ ] Gradio Demo
- [ ] GPT-4o Version.
- [ ] HYDRA with RL


## Installation

### Requirements

- Python >= 3.10
- conda

Please follow the instructions below to install the required packages and set up the environment.

### 1. Clone this repository.
```Bash
git clone https://github.com/ControlNet/HYDRA
```

### 2. Setup conda environment and install dependencies. 
```Bash
bash -i build_env.sh
```

If you meet errors, please consider going through the `build_env.sh` file and install the packages manually.

### 3. Configure the environments

Edit the file `.env` to set the environment variables.

```
OPENAI_API_KEY=your-api-key
# do not change this TORCH_HOME variable
TORCH_HOME=./pretrained_models
```

### 4. Download the pretrained models
Run the scripts to download the pretrained models to the `./pretrained_models` directory. 

```Bash
python download_models.py --base_config <EXP-CONFIG-DIR> --model_config <MODEL-CONFIG-PATH>
```

For example,
```Bash
python download_models.py --base_config ./config/okvqa.yaml --model_config ./configs/model_config_1gpu.yaml
```

## Inference
A worker is required to run the inference. 

```Bash
python run_executor.py --base_config <EXP-CONFIG-DIR> --model_config <MODEL-CONFIG-PATH>
```

### Inference with given one image and prompt
```Bash
python demo_cli.py \
  --image <IMAGE_PATH> \
  --prompt <PROMPT> \
  --base_config <YOUR-CONFIG-DIR> \
  --model_config <MODEL-PATH>
```

### Inference with Gradio GUI
TODO.

### Inference dataset

```Bash
python main.py \
  --data_root <YOUR-DATA-ROOT> \
  --base_config <YOUR-CONFIG-DIR> \
  --model_config <MODEL-PATH>
```

Then the inference results are saved in the `./result` directory for evaluation.

## Evaluation

```Bash
python evaluate.py <RESULT_JSON_PATH> <DATASET_NAME>
```

For example,

```Bash
python evaluate.py result/result_okvqa.jsonl okvqa
```


## Citation
```bibtex
@inproceedings{ke2024hydra,
  title={HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning},
  author={Fucai Ke and Zhixi Cai and Simindokht Jahangard and Weiqing Wang and Pari Delir Haghighi and Hamid Rezatofighi},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```
