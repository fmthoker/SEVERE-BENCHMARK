DATA_DIR="../../data/ava/videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

split_file=$DATA_DIR/../ava_file_names_trainval_v2.1.txt
if [[ ! -f "${split_file}" ]]; then
  echo "${split_file} doesn't exist. Creating it.";
  wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt -O $split_file
fi

for line in $(cat $split_file)
do
  wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done
