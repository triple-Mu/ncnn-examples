#!/bin/bash

name=${1:-yolov8n}

onnx2ncnn \
  ${PWD}/../onnx_models/${name}.onnx \
  ${name}.param \
  ${name}.bin

find ${PWD}/../coco128/ -type f >imagelist.txt
list=${PWD}/imagelist.txt

ncnnoptimize \
  ${name}.param \
  ${name}.bin \
  ${name}-opt.param \
  ${name}-opt.bin \
  0

ncnn2table \
  ${name}-opt.param \
  ${name}-opt.bin \
  ${list} \
  ${name}.table \
  mean=[0,0,0] \
  norm=[0.003921569,0.003921569,0.003921569] \
  shape=[640,640,3] \
  pixel=RGB \
  thread=8 \
  method=kl

ncnn2int8 \
  ${name}-opt.param \
  ${name}-opt.bin \
  ${name}-int8.param \
  ${name}-int8.bin \
  ${name}.table
