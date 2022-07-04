# get conda path
conda_path="$(which conda)"
# get dirname of dirname of conda path
conda_dir="$(dirname $(dirname $conda_path))"
echo $conda_dir

source $conda_dir/etc/profile.d/conda.sh
conda activate slowfast
python -c "import torch; print(torch.__version__)"

echo "Successfully activated conda environment slowfast"