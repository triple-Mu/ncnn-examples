import argparse
import torch
from mmdet.apis import init_detector
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist
from torch.nn import Module
from torch import Tensor

from typing import Tuple, Union

try:
    import onnxsim
except ImportError:
    onnxsim = None
else:
    import onnx


class RTMDet(Module):
    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tuple:
        results = []
        neck_outputs = self.model(inputs)
        for feats in zip(*neck_outputs):
            results.append(torch.cat(feats, 1).permute(0, 2, 3, 1))
        return tuple(results)


class RTMSeg(RTMDet):
    def forward(self, inputs: Tensor) -> Tuple:
        results = []
        neck_outputs = list(self.model(inputs))
        mask = neck_outputs.pop()
        for feats in zip(*neck_outputs):
            results.append(torch.cat(feats, 1).permute(0, 2, 3, 1))
        results.append(mask)
        return tuple(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--task', default='det', help='Task for the model, det or seg')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    args = parser.parse_args()
    assert args.task in ('det', 'seg'), f'input task {args.task} is not supported'
    return args


def build_model_from_cfg(config_path: str, checkpoint_path: str, device: Union[str, int]) -> Module:
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def main():
    args = parse_args()
    config_path = args.config
    checkpoint_path = args.checkpoint
    device = args.device
    out_dir = args.out_dir
    mkdir_or_exist(out_dir)
    onnx_path = f'{out_dir}/rtmdet_{args.task}.onnx'
    img_size = args.img_size
    model = build_model_from_cfg(config_path, checkpoint_path, device)
    if args.task == 'det':
        rtmdet = RTMDet(model)
    else:
        rtmdet = RTMSeg(model)
    rtmdet.eval()

    input = torch.randn(1, 3, *img_size).to(device)

    rtmdet(input)
    torch.onnx.export(
        rtmdet, input,
        onnx_path,
        input_names=['images'],
        opset_version=11)
    if onnxsim is not None:
        onnx_model = onnx.load(onnx_path)
        onnx_model, status = onnxsim.simplify(onnx_model)
        assert status, 'failed to simplify'
        onnx.save(onnx_model, onnx_path)
        print_log(f'Simplified ONNX model saved to {onnx_path}')
    else:
        print_log(f'Exported ONNX model saved to {onnx_path}')


if __name__ == '__main__':
    main()
