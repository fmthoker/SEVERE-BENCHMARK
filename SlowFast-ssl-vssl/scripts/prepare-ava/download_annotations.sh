DATA_DIR="../../data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

# # 2.1
# wget https://research.google.com/ava/download/ava_train_v2.1.csv -P ${DATA_DIR}
# wget https://research.google.com/ava/download/ava_val_v2.1.csv -P ${DATA_DIR}
# wget https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -P ${DATA_DIR}
# wget https://research.google.com/ava/download/ava_train_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
# wget https://research.google.com/ava/download/ava_val_excluded_timestamps_v2.1.csv -P ${DATA_DIR}

# download splits (frame lists)
mkdir -p ../../data/ava/frame_lists/
BASE_URL=https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists
wget $BASE_URL/train.csv -O ../../data/ava/frame_lists/train.csv
wget $BASE_URL/val.csv -O ../../data/ava/frame_lists/val.csv

# # download person boxes
# BASE_URL=https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations
# wget $BASE_URL/ava_train_predicted_boxes.csv -O $DATA_DIR/ava_train_predicted_boxes.csv
# wget $BASE_URL/ava_val_predicted_boxes.csv -O $DATA_DIR/ava_val_predicted_boxes.csv
# wget $BASE_URL/ava_test_predicted_boxes.csv -O $DATA_DIR/ava_test_predicted_boxes.csv

# download annotations for v2.1 and v2.2 together
wget https://dl.fbaipublicfiles.com/pyslowfast/annotation/ava/ava_annotations.tar
tar -xvf ava_annotations.tar
rm ava_annotations.tar
mv ava_annotations/* $DATA_DIR/
rm -rf ava_annotations/
