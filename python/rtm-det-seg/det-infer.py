import numpy as np
import cv2
import argparse
from numpy import ndarray
from typing import List
import math
import ncnn
from config import CLASS_NAMES, CLASS_COLORS, MEAN, STD, sigmoid, path_to_list
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
MAJOR, MINOR = map(int, cv2.__version__.split('.')[:2])
assert MAJOR == 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Image files')
    parser.add_argument('param', help='NCNN param file')
    parser.add_argument('bin', help='NCNN bin file')
    parser.add_argument('--show', action='store_true', help='Show image result')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    args = parser.parse_args()
    return args


def rtmdet_decode(feats: List[ndarray],
                  conf_thres: float,
                  iou_thres: float,
                  num_labels: int = 80,
                  **kwargs):
    proposal_boxes: List[ndarray] = []
    proposal_scores: List[float] = []
    proposal_labels: List[int] = []
    for i, feat in enumerate(feats):
        stride = 8 << i
        score_feat, box_feat = np.split(feat, [
            num_labels,
        ], -1)
        score_feat = sigmoid(score_feat)
        _argmax = score_feat.argmax(-1)
        _max = score_feat.max(-1)
        indices = np.where(_max > conf_thres)
        hIdx, wIdx = indices
        num_proposal = hIdx.size
        if not num_proposal:
            continue

        scores = _max[hIdx, wIdx]
        boxes = box_feat[hIdx, wIdx]
        labels = _argmax[hIdx, wIdx]

        for k in range(num_proposal):
            score = scores[k]
            label = labels[k]

            x0, y0, x1, y1 = boxes[k]

            x0 = wIdx[k] * stride - x0
            y0 = hIdx[k] * stride - y0
            x1 = wIdx[k] * stride + x1
            y1 = hIdx[k] * stride + y1

            w = x1 - x0
            h = y1 - y0

            proposal_scores.append(float(score))
            proposal_boxes.append(
                np.array([x0, y0, w, h], dtype=np.float32))
            proposal_labels.append(int(label))

    if MINOR >= 7:
        indices = cv2.dnn.NMSBoxesBatched(proposal_boxes, proposal_scores, proposal_labels, conf_thres,
                                          iou_thres)
    elif MINOR == 6:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores, conf_thres, iou_thres)
    else:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores, conf_thres, iou_thres).flatten()

    if not indices.size:
        return [], [], []

    nmsd_boxes: List[ndarray] = []
    nmsd_scores: List[float] = []
    nmsd_labels: List[int] = []
    for idx in indices:
        box = proposal_boxes[idx]
        box[2:] = box[:2] + box[2:]
        score = proposal_scores[idx]
        label = proposal_labels[idx]
        nmsd_boxes.append(box)
        nmsd_scores.append(score)
        nmsd_labels.append(label)
    return nmsd_boxes, nmsd_scores, nmsd_labels


def main():
    args = parse_args()
    image_path = Path(args.img)
    bin_path = Path(args.bin)
    param_path = Path(args.param)
    net_h, net_w = args.img_size

    assert image_path.exists(), f'Image {image_path} does not exist'
    assert bin_path.exists(), f'Bin {bin_path} does not exist'
    assert param_path.exists(), f'Param {param_path} does not exist'
    output_names = ['893', '895', '897']

    files = path_to_list(image_path)
    out_dir = None
    if not args.show:
        out_dir = Path(args.out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

    net = ncnn.Net()
    # use gpu or not
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.load_param(str(param_path))
    net.load_model(str(bin_path))

    for file in tqdm(files):
        ex = net.create_extractor()
        img = cv2.imread(file)
        draw_img = img.copy()
        img_w = img.shape[1]
        img_h = img.shape[0]

        ratio_w = img_w / net_w
        ratio_h = img_h / net_h

        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR, img_w, img_h, net_w, net_h
        )
        mat_in.substract_mean_normalize(MEAN, STD)
        ex.input('images', mat_in)

        ret1, mat_out1 = ex.extract(output_names[0])  # stride 8
        assert not ret1, f'extract {output_names[0]} with something wrong!'
        ret2, mat_out2 = ex.extract(output_names[1])  # stride 16
        assert not ret2, f'extract {output_names[1]} with something wrong!'
        ret3, mat_out3 = ex.extract(output_names[2])  # stride 32
        assert not ret3, f'extract {output_names[2]} with something wrong!'
        outputs = [np.array(mat_out1), np.array(mat_out2), np.array(mat_out3)]
        nmsd_boxes, nmsd_scores, nmsd_labels = rtmdet_decode(outputs, 0.25, 0.65)

        for box, score, label in zip(nmsd_boxes, nmsd_scores, nmsd_labels):
            x0, y0, x1, y1 = box
            name = CLASS_NAMES[label]
            box_color = CLASS_COLORS[label]

            x0 = math.floor(min(max(x0 * ratio_w, 1), img_w - 1))
            y0 = math.floor(min(max(y0 * ratio_h, 1), img_h - 1))
            x1 = math.ceil(min(max(x1 * ratio_w, 1), img_w - 1))
            y1 = math.ceil(min(max(y1 * ratio_h, 1), img_h - 1))
            cv2.rectangle(draw_img, (x0, y0), (x1, y1), box_color, 2)
            cv2.putText(draw_img, f'{name}: {score:.2f}',
                        (x0, max(y0 - 5, 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        if args.show:
            cv2.imshow('res', draw_img)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(out_dir / file.name), draw_img)


if __name__ == '__main__':
    main()
