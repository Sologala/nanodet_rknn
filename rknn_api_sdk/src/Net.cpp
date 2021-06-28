#include "Net.hpp"

#include <cassert>
#include <cstring>

#include "rknn_api.h"
#include "testtime.hpp"
namespace RKNN_NET {
Net::Net(const std::string _model_path) : modelPath_(_model_path) {
    printf("Loading~ rknn model\n %s", modelPath_.c_str());
    FILE *fp = fopen(modelPath_.c_str(), "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", modelPath_.c_str());
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);

    int model_len = ftell(fp);
    model_ = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model_, 1, model_len, fp)) {
        printf("fread %s fail!\n", modelPath_.c_str());
        free(model_);
        exit(-1);
    }
    int ret = 0;

    // char device_name[255] = "TDs33101190500578";
    // rknn_init_extend device;
    // device.device_id = device_name;

    // query all avialiable devices
    rknn_devices_id devids;
    ret = rknn_find_devices(&devids);

    printf("n_devices = %d \n", devids.n_devices);
    for (int i = 0; i < devids.n_devices; ++i) {
        printf("%d : type : %s , id %s\n", i, devids.types[i], devids.ids[i]);
    }

    // ret = rknn_init2(&ctx_, model_, model_len, RKNN_FLAG_PRIOR_MEDIUM, &device);
    ret = rknn_init(&ctx_, model_, model_len, RKNN_FLAG_PRIOR_MEDIUM);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(-1);
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        exit(-1);
    }
    printf("inp : %d , output : %d\n", io_num.n_input, io_num.n_output);
    inputs_attr = new rknn_tensor_attr[io_num.n_input];
    printf("-------------[input %d ]-------------\n", io_num.n_input);
    for (int i = 0, sz = io_num.n_input; i < sz; ++i) {
        inputs_attr[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inputs_attr[i], sizeof(inputs_attr[i]));
        if (ret < 0) {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        printf("%d : (", i);
        for (int j = 0; j < inputs_attr[i].n_dims; ++j) {
            if (j) printf(", ");
            printf("%d", inputs_attr[i].dims[j]);
        }
        printf(") ");
    }
    printf("\n-------------[output %d ]-------------\n", io_num.n_output);
    outputs_attr = new rknn_tensor_attr[io_num.n_output];
    for (int i = 0, sz = io_num.n_output; i < sz; ++i) {
        outputs_attr[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &outputs_attr[i], sizeof(outputs_attr[i]));
        if (ret < 0) {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        printf("%d : (", i);
        for (int j = 0; j < outputs_attr[i].n_dims; ++j) {
            if (j) printf(", ");
            printf("%d", outputs_attr[i].dims[j]);
        }
        printf(") ");
    }
    printf("\n");

    n_inputs_ = io_num.n_input;
    n_outputs_ = io_num.n_output;
}

#define __DO_TESTTIME__

void Net::Forward(std::vector<void *> _datas) {
#ifdef __DO_TESTTIME__
    TestTimeVar;
    TestTimeTic;
#endif
    assert(_datas.size() == n_inputs_);
    for (int i = 0; i < n_inputs_; ++i) {
        inputs_[i].buf = _datas[i];
        inputs_[i].pass_through = false;
    }

    int ret = rknn_inputs_set(ctx_, n_inputs_, inputs_);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        exit(-1);
    }
#ifdef __DO_TESTTIME__
    TestTimeToc("set input : ");
    TestTimeTic;
#endif
    ret = rknn_run(ctx_, NULL);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        exit(-1);
    }

    ret = rknn_outputs_get(ctx_, n_outputs_, outputs_, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        exit(-1);
    }
#ifdef __DO_TESTTIME__
    TestTimeToc("inference : ");
#endif
}

void Net::Input_Output_Configuration() {
    // 配置 input 与 output 数组
    inputs_ = new rknn_input[n_inputs_];
    outputs_ = new rknn_output[n_outputs_];

    for (int i = 0; i < n_inputs_; ++i) {
        inputs_[i].index = i;
        inputs_[i].buf = NULL;
        inputs_[i].size = inputs_attr[i].n_elems * sizeof(char);
        inputs_[i].pass_through = false;
        inputs_[i].type = RKNN_TENSOR_UINT8;
        inputs_[i].fmt = RKNN_TENSOR_NHWC;
    }

    outputs_buffer = new float *[n_outputs_];

    for (int i = 0; i < n_outputs_; ++i) {
        outputs_[i].want_float = true;
        outputs_[i].is_prealloc = true;
        outputs_[i].index = i;
        outputs_[i].size = outputs_attr[i].n_elems * sizeof(float);
        outputs_buffer[i] = new float[outputs_[i].size];
        outputs_[i].buf = (void *)outputs_buffer[i];
        memset(outputs_[i].buf, 0, sizeof(outputs_[i].buf));
    }
}

Net::~Net() {
    printf("Free net~~~ \n");
    // free output buffer ;
    for (int i = 0; i < n_outputs_; ++i) {
        delete[] outputs_buffer[i];
    }
    if (outputs_buffer) delete[] outputs_buffer;
    if (outputs_) delete[] outputs_;
    if (inputs_) delete[] inputs_;
    if (inputs_attr) delete[] inputs_attr;
    if (outputs_attr) delete[] outputs_attr;
    rknn_destroy(ctx_);
    // delete model
    free(model_);
}
}  // namespace RKNN_NET