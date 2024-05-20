#!/usr/bin/env bash

./build.sh

docker save hanseg2023algorithm_dmx | gzip -c > HanSeg2023AlgorithmDMX.tar.gz
