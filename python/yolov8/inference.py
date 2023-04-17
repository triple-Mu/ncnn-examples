import ncnn
import numpy as np
import cv2
import math
import random
from numpy import ndarray
from typing import List, Tuple
from enum import Enum
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

MAJOR, MINOR = map(int, cv2.__version__.split('.')[:2])
assert MAJOR == 4

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


class TASK_TYPE(Enum):
    DET = 'detect'
    SEG = 'segment'
    POSE = 'pose'


def path_to_list(path: str):
    path = Path(path)
    if path.is_file() and path.suffix in IMG_EXTENSIONS:
        res_list = [str(path.absolute())]
    elif path.is_dir():
        res_list = [
            str(p.absolute()) for p in path.iterdir()
            if p.suffix in IMG_EXTENSIONS
        ]
    else:
        raise RuntimeError
    return res_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('param', type=str, help='Param file')
    parser.add_argument('bin', type=str, help='Param file')
    parser.add_argument('--type', type=str, help='Task type')
    parser.add_argument('--input-size', type=int, default=640, help='Input size')

    parser.add_argument(
        '--out-dir', default='./output', type=str, help='Path to output file')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.25, help='Bbox score threshold')
    parser.add_argument(
        '--iou-thr', type=float, default=0.65, help='Bbox iou threshold')
    args = parser.parse_args()
    return args


random.seed(0)

CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

MASK_COLORS = np.array([[255, 56, 56], [255, 157, 151], [255, 112, 31],
                        [255, 178, 29], [207, 210, 49], [72, 249, 10],
                        [146, 204, 23], [61, 219, 134], [26, 147, 52],
                        [0, 212, 187], [44, 153, 168], [0, 194, 255],
                        [52, 69, 147], [100, 115, 255], [0, 24, 236],
                        [132, 56, 255], [82, 0, 133], [203, 56, 255],
                        [255, 149, 200], [255, 55, 199]], dtype=np.uint8)
KPS_COLORS = [[0, 255, 0], [0, 255, 0], [0, 255, 0],
              [0, 255, 0], [0, 255, 0], [255, 128, 0],
              [255, 128, 0], [255, 128, 0], [255, 128, 0],
              [255, 128, 0], [255, 128, 0], [51, 153, 255],
              [51, 153, 255], [51, 153, 255], [51, 153, 255],
              [51, 153, 255], [51, 153, 255]]

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13],
            [12, 13], [6, 12], [7, 13], [6, 7],
            [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4],
            [3, 5], [4, 6], [5, 7]]

LIMB_COLORS = [[51, 153, 255], [51, 153, 255], [51, 153, 255],
               [51, 153, 255], [255, 51, 255], [255, 51, 255],
               [255, 51, 255], [255, 128, 0], [255, 128, 0],
               [255, 128, 0], [255, 128, 0], [255, 128, 0],
               [0, 255, 0], [0, 255, 0], [0, 255, 0],
               [0, 255, 0], [0, 255, 0], [0, 255, 0],
               [0, 255, 0]]


def softmax(x: ndarray, axis: int = -1) -> ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)
    return y


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


def postprocess(feats: List[ndarray],
                task: TASK_TYPE,
                conf_thres: float = 0.25,
                reg_max: int = 16) -> Tuple:
    dfl = np.arange(0, reg_max, dtype=np.float32)
    scores_pro = []
    boxes_pro = []
    labels_pro = []
    if task == TASK_TYPE.SEG:
        mcoefs_pro = []
    elif task == TASK_TYPE.POSE:
        kpss_pro = []

    for i, feat in enumerate(feats):
        stride = 8 << i
        if task == TASK_TYPE.DET:
            score_feat, box_feat = np.split(feat, [80, ], -1)
        elif task == TASK_TYPE.SEG:
            score_feat, box_feat, mcoef_feat = np.split(feat, [80, 80 + 64], -1)
        elif task == TASK_TYPE.POSE:
            score_feat, box_feat, kps_feat = np.split(feat, [1, 1 + 64], -1)

        score_feat = sigmoid(score_feat)
        if task == TASK_TYPE.DET or task == TASK_TYPE.SEG:
            _argmax = score_feat.argmax(-1)
            _max = score_feat.max(-1)
        elif task == TASK_TYPE.POSE:
            _max = score_feat.squeeze(-1)

        indices = np.where(_max > conf_thres)
        hIdx, wIdx = indices
        num_proposal = hIdx.size
        if not num_proposal:
            continue

        scores = _max[hIdx, wIdx]
        boxes = box_feat[hIdx, wIdx].reshape(-1, 4, reg_max)
        boxes = softmax(boxes, -1) @ dfl
        if task == TASK_TYPE.DET or task == TASK_TYPE.SEG:
            argmax = _argmax[hIdx, wIdx]
        if task == TASK_TYPE.SEG:
            mcoefs = mcoef_feat[hIdx, wIdx]
        elif task == TASK_TYPE.POSE:
            kpss = kps_feat[hIdx, wIdx]

        for k in range(num_proposal):
            h, w = hIdx[k], wIdx[k]
            score = scores[k]
            x0, y0, x1, y1 = boxes[k]

            x0 = (w + 0.5 - x0) * stride
            y0 = (h + 0.5 - y0) * stride
            x1 = (w + 0.5 + x1) * stride
            y1 = (h + 0.5 + y1) * stride

            if task == TASK_TYPE.DET or task == TASK_TYPE.SEG:
                clsid = argmax[k]
            elif task == TASK_TYPE.POSE:
                clsid = 0

            if task == TASK_TYPE.SEG:
                mcoef = mcoefs[k]
            elif task == TASK_TYPE.POSE:
                kps = kpss[k].reshape(-1, 3)
                kps[:, :1] = (kps[:, :1] * 2. + w) * stride
                kps[:, 1:2] = (kps[:, 1:2] * 2. + h) * stride
                kps[:, 2:3] = sigmoid(kps[:, 2:3])

            scores_pro.append(float(score))
            boxes_pro.append(np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32))
            labels_pro.append(clsid)
            if task == TASK_TYPE.SEG:
                mcoefs_pro.append(mcoef)
            elif task == TASK_TYPE.POSE:
                kpss_pro.append(kps)

    if task == TASK_TYPE.DET:
        results = (boxes_pro, scores_pro, labels_pro)
    elif task == TASK_TYPE.SEG:
        results = (boxes_pro, scores_pro, labels_pro, mcoefs_pro)
    elif task == TASK_TYPE.POSE:
        results = (boxes_pro, scores_pro, labels_pro, kpss_pro)

    return results


if __name__ == '__main__':
    # modify output name list by onnx output names
    OUTPUT_NAMES = {
        TASK_TYPE.DET: ['330', '347', '364'],
        TASK_TYPE.SEG: ['374', '399', '424', '349'],
        TASK_TYPE.POSE: ['356', '381', '406']
    }
    args = parse_args()
    out_dir = Path(args.out_dir)
    task_type = TASK_TYPE(args.type.lower())
    output_names = OUTPUT_NAMES[task_type]

    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = path_to_list(args.img)

    net = ncnn.Net()
    # use gpu or not
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 16

    net.load_param(args.param)
    net.load_model(args.bin)

    for file in tqdm(files):
        ex = net.create_extractor()
        img = cv2.imread(file)
        img_w = img.shape[1]
        img_h = img.shape[0]

        w = img_w
        h = img_h
        scale = 1.0
        if w > h:
            scale = float(args.input_size) / w
            w = args.input_size
            h = int(h * scale)
        else:
            scale = float(args.input_size) / h
            h = args.input_size
            w = int(w * scale)

        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, w, h
        )

        wpad = (w + 31) // 32 * 32 - w
        hpad = (h + 31) // 32 * 32 - h
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad // 2,
            hpad - hpad // 2,
            wpad // 2,
            wpad - wpad // 2,
            ncnn.BorderType.BORDER_CONSTANT,
            114.0,
        )

        dw = wpad // 2
        dh = hpad // 2

        mat_in_pad.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex.input('images', mat_in_pad)

        ret1, mat_out1 = ex.extract(output_names[0])  # stride 8
        assert not ret1, f'extract {output_names[0]} with something wrong!'
        ret2, mat_out2 = ex.extract(output_names[1])  # stride 16
        assert not ret2, f'extract {output_names[1]} with something wrong!'
        ret3, mat_out3 = ex.extract(output_names[2])  # stride 32
        assert not ret3, f'extract {output_names[2]} with something wrong!'

        if task_type == TASK_TYPE.SEG:
            ret4, mat_out4 = ex.extract(output_names[3])  # masks
            assert not ret4, f'extract {output_names[3]} with something wrong!'
            proto = np.array(mat_out4)

        outputs = [np.array(mat_out1), np.array(mat_out2), np.array(mat_out3)]
        results = postprocess(outputs, task_type, args.score_thr, 16)

        if task_type == TASK_TYPE.DET:
            boxes_pro, scores_pro, labels_pro = results

        elif task_type == TASK_TYPE.SEG:
            boxes_pro, scores_pro, labels_pro, mcoefs_pro = results

        elif task_type == TASK_TYPE.POSE:
            boxes_pro, scores_pro, labels_pro, kpss_pro = results

        else:
            raise NotImplementedError

        if MINOR >= 7:
            indices = cv2.dnn.NMSBoxesBatched(boxes_pro, scores_pro, labels_pro, args.score_thr,
                                              args.iou_thr)
        elif MINOR == 6:
            indices = cv2.dnn.NMSBoxes(boxes_pro, scores_pro, args.score_thr, args.iou_thr)
        else:
            indices = cv2.dnn.NMSBoxes(boxes_pro, scores_pro, args.score_thr, args.iou_thr).flatten()

        if task_type == TASK_TYPE.SEG:
            mask_img = img.copy()

        for idx in indices:
            box = boxes_pro[idx]
            score = scores_pro[idx]
            clsid = labels_pro[idx]

            color = CLASS_COLORS[clsid]
            box[2:] = box[:2] + box[2:]
            x0, y0, x1, y1 = box

            # clip feature
            x0 = min(max(x0, 1), w - 1)
            y0 = min(max(y0, 1), h - 1)
            x1 = min(max(x1, 1), w - 1)
            y1 = min(max(y1, 1), h - 1)

            if task_type == TASK_TYPE.SEG:
                _x0, _y0, _x1, _y1 = math.floor(x0 / 4), math.floor(y0 / 4), math.ceil(x1 / 4), math.ceil(y1 / 4)

            x0 = (x0 - dw) / scale
            y0 = (y0 - dh) / scale
            x1 = (x1 - dw) / scale
            y1 = (y1 - dh) / scale

            # clip image
            x0 = min(max(x0, 1), img_w - 1)
            y0 = min(max(y0, 1), img_h - 1)
            x1 = min(max(x1, 1), img_w - 1)
            y1 = min(max(y1, 1), img_h - 1)

            x0, y0, x1, y1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)

            if task_type == TASK_TYPE.SEG:
                mcoef = mcoefs_pro[idx]
                mcolor = MASK_COLORS[clsid % len(MASK_COLORS)]
                _proto = proto[:, _y0:_y1, _x0:_x1]
                mask = sigmoid(mcoef @ _proto.reshape(32, -1))
                mask = mask.reshape(_y1 - _y0, _x1 - _x0)
                mask = cv2.resize(mask, (x1 - x0, y1 - y0)) > 0.5
                mask = mask[..., np.newaxis]
                mask_img[y0:y1, x0:x1] = (~mask * mask_img[y0:y1, x0:x1] + mask * mcolor).astype(np.uint8)


            elif task_type == TASK_TYPE.POSE:
                kps = kpss_pro[idx]
                for i in range(19):
                    if i < 17:
                        px, py, ps = kps[i]
                        px = round((px - dw) / scale)
                        py = round((py - dh) / scale)
                        if ps > 0.5:
                            kcolor = KPS_COLORS[i]
                            cv2.circle(img, (px, py), 5, kcolor, -1)
                    xi, yi = SKELETON[i]
                    pos1_s = kps[xi - 1][2]
                    pos2_s = kps[yi - 1][2]
                    if pos1_s > 0.5 and pos2_s > 0.5:
                        limb_color = LIMB_COLORS[i]
                        pos1_x = round((kps[xi - 1][0] - dw) / scale)
                        pos1_y = round((kps[xi - 1][1] - dw) / scale)

                        pos2_x = round((kps[yi - 1][0] - dw) / scale)
                        pos2_y = round((kps[yi - 1][1] - dw) / scale)

                        cv2.line(img, (pos1_x, pos1_y), (pos2_x, pos2_y), limb_color, 2)

            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(img, f'{CLASS_NAMES[clsid]}: {score:.2f}', (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2)

        if task_type == TASK_TYPE.SEG:
            img = cv2.addWeighted(img, 0.5, mask_img, 0.8, 0)
        if args.show:
            cv2.imshow('result', img)
            key = cv2.waitKey(0)
            if key & 0XFF == ord('q'):
                break
        else:
            cv2.imwrite(f'out_dir/{Path(file).name}', img)
