#!/bin/bash

name=${1:-rtmdet_det}

onnx2ncnn \
  ${PWD}/../onnx_models/${name}.onnx \
  ${name}.param \
  ${name}.bin

find ${PWD}/../../../coco128/ -type f >imagelist.txt
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
  mean=[103.53,116.28,123.675] \
  norm=[0.01742919,0.017507,0.01712461] \
  shape=[640,640,3] \
  pixel=BGR \
  thread=8 \
  method=kl

ncnn2int8 \
  ${name}-opt.param \
  ${name}-opt.bin \
  ${name}-int8.param \
  ${name}-int8.bin \
  ${name}.table
