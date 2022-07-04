# setup videos that failed in the first go
video_id=I8j6Xq2B5ys

# download
wget https://s3.amazonaws.com/ava-dataset/trainval/$video_id.mkv ../../data/ava/videos/

# cut into 15min chunk
cd scripts/prepare-ava/
bash cut_videos.sh

# extract frames
OUT_DATA_DIR="../../data/ava/frames"
video=../../data/ava/videos_15min/I8j6Xq2B5ys.mp4
video_name=${video##*/}
video_name=${video_name::-4}
out_video_dir=${OUT_DATA_DIR}/${video_name}/
out_name="${out_video_dir}/${video_name}_%06d.jpg"
ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"