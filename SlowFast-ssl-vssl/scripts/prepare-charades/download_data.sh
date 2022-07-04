DATA_DIR="../../data/charades"
mkdir -p ${DATA_DIR}

# download RGB frames
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_rgb.tar -P ${DATA_DIR}/
tar -xvf ${DATA_DIR}/Charades_v1_rgb.tar --directory ${DATA_DIR}