# Usage: bash create_conda_env.sh
# Outcome: Creates a conda environment `slowfast` for the project

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
REMOVE_ALL=$(tput sgr0)

echo "::: $YELLOW Checking conda installation ... $REMOVE_ALL"

if ! command -v conda --version &> /dev/null
then
    echo "::: ERROR: conda is not installed"
    exit 1
else
    echo "::: conda is installed with version $(conda --version)"
fi

ENV_NAME=$1
if [ -z "$ENV_NAME" ]
then
    ENV_NAME="slowfast"
fi

echo "::: $YELLOW Creating conda environment $ENV_NAME ... $REMOVE_ALL"

conda create -y -n $ENV_NAME python=3.9

# initialize conda to be able to activate the environment
# get conda path
conda_path="$(which conda)"
# get dirname of dirname of conda path
conda_dir="$(dirname $(dirname $conda_path))"
echo "Conda home: $conda_dir"
source $conda_dir/etc/profile.d/conda.sh

# activate the environment
conda activate $ENV_NAME

echo "::: $YELLOW Installing torch-packages ... $REMOVE_ALL"
# check your apt pytorch version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# install other packages
echo "::: $YELLOW Checking GCC version: $(gcc --version) ... $REMOVE_ALL"
echo "::: $YELLOW Installing other packages ... $REMOVE_ALL"
pip install simplejson
conda install av -y -c conda-forge
conda install -y -c iopath iopath
pip install psutil
pip install opencv-python
pip install tensorboard
conda install -y -c conda-forge moviepy
pip install pytorchvideo
pip install 'git+https://github.com/facebookresearch/fairscale'
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' \
    'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
pip install scikit-learn
pip install pyhocon
pip install ipdb

echo "::: $GREEN Done! $REMOVE_ALL"

echo "::: $YELLOW Testing the environment: $REMOVE_ALL"
conda activate $ENV_NAME
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import torchvision; print('Torchvision:', torchvision.__version__)"
python -c "import torchaudio; print('Torchaudio:', torchaudio.__version__)"
python -c "import detectron2; print('detectron2:', detectron2.__version__)"