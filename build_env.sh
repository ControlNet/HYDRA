cp .env.example .env
# then modify the openai api in the .env file

conda create -n hydra python=3.11 -y
conda activate hydra

# install gcc, gxx compiler
conda install gcc=9 gxx=9 cxx-compiler -y -c conda-forge

# install pytorch, cuda and other dependencies
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-11.8.0::cuda -y
pip install -r requirements.txt

# install GLIP
cd module_repos/GLIP && CUDA_HOME=$CONDA_PREFIX python setup.py clean --all build develop --user && cd ../..

# install grounded sam
cd module_repos/Grounded-Segment-Anything
# install sam
AM_I_DOCKER=False BUILD_WITH_CUDA=True CUDA_HOME=$CONDA_PREFIX pip install -e ./segment_anything
# install grounding dino
AM_I_DOCKER=False BUILD_WITH_CUDA=True CUDA_HOME=$CONDA_PREFIX pip install --no-build-isolation -e ./GroundingDINO
cd ../..

# install llava
AM_I_DOCKER=False BUILD_WITH_CUDA=True CUDA_HOME=$CONDA_PREFIX pip install -e module_repos/LLaVA

pip install "scipy==1.10.*"  # fix scipy
