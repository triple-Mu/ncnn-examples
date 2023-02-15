#include "layer.h"
#include "net.h"

#include "opencv2/opencv.hpp"

#include <float.h>
#include <stdio.h>
#include <vector>

#define MAX_STRIDE 32 // if yolov5-p6 model modify to 64

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};

static void generate_proposals(
	const ncnn::Mat& anchors,
	const int stride,
	const ncnn::Mat& feat_blob,
	const float prob_threshold,
	std::vector<Object>& objects
)
{
	const int num_w = feat_blob.w;
	const int num_grid_y = feat_blob.c;
	const int num_grid_x = feat_blob.h;

	const int num_anchors = anchors.w / 2;
	const int walk = num_w / num_anchors;
	const int num_class = walk - 5;

	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{

			const float* matat = feat_blob.channel(i).row(j);

			for (int k = 0; k < num_anchors; k++)
			{
				const float anchor_w = anchors[k * 2];
				const float anchor_h = anchors[k * 2 + 1];
				const float* ptr = matat + k * walk;
				float box_confidence = ptr[4];
				if (box_confidence >= prob_threshold)
				{
					// find class index with max class score
					int class_index = 0;
					float class_score = -FLT_MAX;
					for (int c = 0; c < num_class; c++)
					{
						float score = ptr[5 + c];
						if (score > class_score)
						{
							class_index = c;
							class_score = score;
						}
						float confidence = box_confidence * class_score;

						if (confidence >= prob_threshold)
						{
							float dx = ptr[0];
							float dy = ptr[1];
							float dw = ptr[2];
							float dh = ptr[3];

							float pb_cx = (dx * 2.f - 0.5f + j) * stride;
							float pb_cy = (dy * 2.f - 0.5f + i) * stride;

							float pb_w = powf(dw * 2.f, 2) * anchor_w;
							float pb_h = powf(dh * 2.f, 2) * anchor_h;

							float x0 = pb_cx - pb_w * 0.5f;
							float y0 = pb_cy - pb_h * 0.5f;
							float x1 = pb_cx + pb_w * 0.5f;
							float y1 = pb_cy + pb_h * 0.5f;

							Object obj;
							obj.rect.x = x0;
							obj.rect.y = y0;
							obj.rect.width = x1 - x0;
							obj.rect.height = y1 - y0;
							obj.label = class_index;
							obj.prob = confidence;

							objects.push_back(obj);

						}
					}
				}
			}
		}
	}
}

static float clamp(
	float val,
	float min = 0.f,
	float max = 1280.f
)
{
	return val > min ? (val < max ? val : max) : min;
}
static void non_max_suppression(
	std::vector<Object>& proposals,
	std::vector<Object>& results,
	int orin_h,
	int orin_w,
	float dh = 0,
	float dw = 0,
	float ratio_h = 1.0f,
	float ratio_w = 1.0f,
	float conf_thres = 0.25f,
	float iou_thres = 0.65f
)
{
	results.clear();
	std::vector<cv::Rect> bboxes;
	std::vector<float> scores;
	std::vector<int> labels;
	std::vector<int> indices;

	for (auto& pro : proposals)
	{
		bboxes.push_back(pro.rect);
		scores.push_back(pro.prob);
		labels.push_back(pro.label);
	}

	cv::dnn::NMSBoxes(
		bboxes,
		scores,
		conf_thres,
		iou_thres,
		indices
	);

	for (auto i : indices)
	{
		auto& bbox = bboxes[i];
		float x0 = bbox.x;
		float y0 = bbox.y;
		float x1 = bbox.x + bbox.width;
		float y1 = bbox.y + bbox.height;
		float& score = scores[i];
		int& label = labels[i];

		x0 = (x0 - dw) / ratio_w;
		y0 = (y0 - dh) / ratio_h;
		x1 = (x1 - dw) / ratio_w;
		y1 = (y1 - dh) / ratio_h;

		x0 = clamp(x0, 0.f, orin_w);
		y0 = clamp(y0, 0.f, orin_h);
		x1 = clamp(x1, 0.f, orin_w);
		y1 = clamp(y1, 0.f, orin_h);

		Object obj;
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = score;
		obj.label = label;
		results.push_back(obj);
	}
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
	ncnn::Net yolov5;

	yolov5.opt.use_vulkan_compute = true;
	// yolov5.opt.use_bf16_storage = true;

	// original pretrained model from https://github.com/ultralytics/yolov5
	// the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
	if (yolov5.load_param("yolov5s.ncnn.param"))
		exit(-1);
	if (yolov5.load_model("yolov5s.ncnn.bin"))
		exit(-1);

	const int target_size = 640;
	const float prob_threshold = 0.25f;
	const float nms_threshold = 0.45f;

	int img_w = bgr.cols;
	int img_h = bgr.rows;

	// letterbox pad to multiple of MAX_STRIDE
	int w = img_w;
	int h = img_h;
	float scale = 1.f;
	if (w > h)
	{
		scale = (float)target_size / w;
		w = target_size;
		h = h * scale;
	}
	else
	{
		scale = (float)target_size / h;
		h = target_size;
		w = w * scale;
	}

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

	// pad to target_size rectangle
	// yolov5/utils/datasets.py letterbox
	int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
	int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

	int top = hpad / 2;
	int bottom = hpad - hpad / 2;
	int left = wpad / 2;
	int right = wpad - wpad / 2;

	ncnn::Mat in_pad;
	ncnn::copy_make_border(in,
		in_pad,
		top,
		bottom,
		left,
		right,
		ncnn::BORDER_CONSTANT,
		114.f);

	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	in_pad.substract_mean_normalize(0, norm_vals);

	ncnn::Extractor ex = yolov5.create_extractor();

	ex.input("in0", in_pad);

	std::vector<Object> proposals;

	// anchor setting from yolov5/models/yolov5s.yaml

	// stride 8
	{
		ncnn::Mat out;
		ex.extract("out0", out);

		ncnn::Mat anchors(6);
		anchors[0] = 10.f;
		anchors[1] = 13.f;
		anchors[2] = 16.f;
		anchors[3] = 30.f;
		anchors[4] = 33.f;
		anchors[5] = 23.f;

		std::vector<Object> objects8;
		generate_proposals(anchors, 8, out, prob_threshold, objects8);

		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
	}

	// stride 16
	{
		ncnn::Mat out;

		ex.extract("out1", out);

		ncnn::Mat anchors(6);
		anchors[0] = 30.f;
		anchors[1] = 61.f;
		anchors[2] = 62.f;
		anchors[3] = 45.f;
		anchors[4] = 59.f;
		anchors[5] = 119.f;

		std::vector<Object> objects16;
		generate_proposals(anchors, 16, out, prob_threshold, objects16);

		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
	}

	// stride 32
	{
		ncnn::Mat out;

		ex.extract("out2", out);

		ncnn::Mat anchors(6);
		anchors[0] = 116.f;
		anchors[1] = 90.f;
		anchors[2] = 156.f;
		anchors[3] = 198.f;
		anchors[4] = 373.f;
		anchors[5] = 326.f;

		std::vector<Object> objects32;
		generate_proposals(anchors, 32, out, prob_threshold, objects32);

		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
	}

	non_max_suppression(proposals, objects,
		img_h, img_w, hpad / 2, wpad / 2,
		scale, scale, prob_threshold, nms_threshold);
	return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	static const char* class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	cv::Mat image = bgr.clone();

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = obj.rect.x;
		int y = obj.rect.y - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > image.cols)
			x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}

	cv::imshow("image", image);
	cv::waitKey(0);
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
		return -1;
	}

	const char* imagepath = argv[1];

	cv::Mat m = cv::imread(imagepath, 1);
	if (m.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	}

	std::vector<Object> objects;
	detect_yolov5(m, objects);

	draw_objects(m, objects);

	return 0;
}