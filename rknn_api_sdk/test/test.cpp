#include <opencv2/core.hpp>

#include "Net.hpp"
#include "testtime.hpp"
using namespace std;

int main(int argc, const char** argv) {
    const int input_shape[4] = {1, 3, 320, 320};
    TestTimeVar;
    TestTimeTic;

    for (int i = 0; i < 10000000; ++i) {
        TestTimeTic;
        RKNN_NET::Net net("../nanodet.rknn");
        net.Input_Output_Configuration();
        TestTimeToc("load");
        cv::Mat img = cv::Mat::zeros(320, 320, CV_8UC3);
        net.Forward({img.data});
        TestTimeToc("run");
    }
    return 0;
}
