CKPT_DIR="../../checkpoints/kinetics400/"
mkdir -p $CKPT_DIR
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_8x8_R50.pkl -O $CKPT_DIR/SLOWONLY_8x8_R50.pkl
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_4x16_R50.pkl -O $CKPT_DIR/SLOWONLY_4x16_R50.pkl
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl -O $CKPT_DIR/SLOWFAST_8x8_R50.pkl