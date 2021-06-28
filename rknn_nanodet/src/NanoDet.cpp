#include "NanoDet.hpp"

#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Net.hpp"
using namespace std;

inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

NanoDet::~NanoDet() {}

NanoDet::NanoDet(const string _modelpath) : RKNN_NET::Net(_modelpath)
{
    // do s
    Input_Output_Configuration();
}

std::vector<std::vector<BoxInfo>> NanoDet::detect(const cv::Mat _img)
{
    cv::Mat resizeed_img;
    float fx = 1, fy = 1;
    if (_img.rows != intput_w || _img.cols != intput_h)
    {
        cv::resize(_img, resizeed_img, cv::Size(intput_w, intput_h));
        fx = (_img.cols * 1.0) / intput_w;
        fy = (_img.rows * 1.0) / intput_h;
    }
    else
    {
        resizeed_img = _img;
    }
    RKNN_NET::Net::Forward({resizeed_img.data});
    std::vector<std::vector<BoxInfo>> res;
    res.resize(labels_.size());
    for (int i = 0; i < stride_.size(); ++i)
    {
        const int idx_class = i;
        const int idx_bbx = i + 3;

        float *bbx_pred = (float *)outputs_[idx_bbx].buf;
        float *class_pred = (float *)outputs_[idx_class].buf;
        // get output
        this->decode_infer(class_pred, bbx_pred, stride_[i], score_threshold_, res);
    }

    for (int i = 0; i < labels_.size(); ++i)
    {
        nms(res[i], 0.5);
        for (auto &_bbx : res[i])
        {
            _bbx.x1 *= fx;
            _bbx.x2 *= fx;
            _bbx.y1 *= fy;
            _bbx.y2 *= fy;
        }
    }
    return res;
}

void NanoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b)
              { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

void NanoDet::Input_Output_Configuration()
{
    // 配置 input 与 output 数组
    inputs_ = new rknn_input[n_inputs_];
    outputs_ = new rknn_output[n_outputs_];

    for (int i = 0; i < n_inputs_; ++i)
    {
        inputs_[i].index = i;
        inputs_[i].buf = NULL;
        inputs_[i].size = inputs_attr[i].n_elems * sizeof(char);
        inputs_[i].pass_through = false;
        inputs_[i].type = RKNN_TENSOR_UINT8;
        inputs_[i].fmt = RKNN_TENSOR_NHWC;
    }

    outputs_buffer = new float *[n_outputs_];

    for (int i = 0; i < n_outputs_; ++i)
    {
        outputs_[i].want_float = true;
        outputs_[i].is_prealloc = true;
        outputs_[i].index = i;
        outputs_[i].size = outputs_attr[i].n_elems * sizeof(float);
        outputs_buffer[i] = new float[outputs_[i].size];
        outputs_[i].buf = (void *)outputs_buffer[i];
        memset(outputs_[i].buf, 0, sizeof(outputs_[i].buf));
    }
}

void NanoDet::decode_infer(float *cls_pred, float *&dis_pred, int stride, float threshold,
                           std::vector<std::vector<BoxInfo>> &results)
{
    int feature_h = 320 / stride;
    int feature_w = 320 / stride;
    // cv::Mat debug_heatmap = cv::Mat::zeros(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < feature_h * feature_w; idx++)
    {
        int row = idx / feature_w;
        int col = idx % feature_w;

        float score = 0;
        int cur_label = 0;

        // 找一个最大的score
        for (int label = 0, num_class_ = labels_.size(); label < num_class_; label++)
        {
            if (cls_pred[idx * num_class_ + label] > score)
            {
                score = cls_pred[idx * num_class_ + label];
                cur_label = label;
            }
        }
        if (score > threshold)
        {
            if (cur_label == 0 || 1)
            {
                // std::cout << row << "," << col << " label:" << cur_label << " score:" << score << std::endl;
                const float *bbox_pred = dis_pred + idx * (reg_max_ + 1) * 4;
                results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
            }

            // debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
        }
    }
    // cv::imshow("debug", debug_heatmap);
    // cv::waitKey(0);
}

BoxInfo NanoDet::disPred2Bbox(const float *&bbox_pred, int label, float score, int x, int y, int stride)
{
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float *dis_after_sm = new float[reg_max_ + 1];
        activation_function_softmax(bbox_pred + i * (reg_max_ + 1), dis_after_sm, reg_max_ + 1);
        for (int j = 0; j < reg_max_ + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        // std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], float(intput_w));
    float ymax = (std::min)(ct_y + dis_pred[3], float(intput_h));

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}
