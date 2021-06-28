#ifndef NET_H
#define NET_H
#include <string>
#include <vector>

#include "rknn_api.h"

#pragma once
namespace RKNN_NET
{
    class Net
    {
    public:
        Net(const std::string _model_path);
        virtual ~Net();
        virtual void Input_Output_Configuration();
        void Forward(std::vector<void *> _datas);

    protected:
        void *model_;
        rknn_context ctx_;
        std::string modelPath_;
        int n_inputs_, n_outputs_;

        rknn_input *inputs_;
        rknn_tensor_attr *inputs_attr;
        rknn_output *outputs_;
        rknn_tensor_attr *outputs_attr;
        float **outputs_buffer;
    };
} // namespace RKNN_NET
#endif