#ifndef NANODET_H
#define NANODET_H
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

#include "Net.hpp"
#include "rknn_api.h"
#pragma once

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class NanoDet : public RKNN_NET::Net
{
public:
    NanoDet(const std::string _modelpath);
    ~NanoDet();
    std::vector<std::vector<BoxInfo>> detect(const cv::Mat _img);
    void Input_Output_Configuration();
    void decode_infer(float *cls_pred, float *&dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>> &results);

    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride);

    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

    std::vector<std::string> labels_{
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"};

private:
    std::vector<int> stride_ = {8, 16, 32};
    const int reg_max_ = 7;
    const int intput_w = 320;
    const int intput_h = 320;
    const float score_threshold_ = 0.35;
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
};
#endif