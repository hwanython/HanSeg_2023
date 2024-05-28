#!/usr/bin/env bash
# 현재 디렉토리 서칭
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH 
OUTDIR=$SCRIPTPATH/output

./build.sh

# Ensure output directory exists and set permissions
mkdir -p $OUTDIR
chmod -R 777 $OUTDIR

# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="30g"

# create output dir if it does not exist
if [ ! -d $OUTDIR ]; then
  mkdir $OUTDIR;
fi
echo "starting docker"
docker run --rm --name hanseg_algorithm_container \
    --memory="${MEM_LIMIT}" \
    --memory-swap="${MEM_LIMIT}" \
    --network="none" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    --shm-size="4g" \
    --pids-limit="512" \
    --gpus="all" \
    -v $SCRIPTPATH/input/:/input/ \
    -v $SCRIPTPATH/output/:/output \
    -e LOCAL_USER_ID=$(id -u) \
    -e LOCAL_GROUP_ID=$(id -g) \
    -it --entrypoint bash \
    hanseg2023algorithm_dmx:jhhan

echo "docker done"
