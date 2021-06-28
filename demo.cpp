#include "NanoDet.hpp"
#include <string>
#include "testtime.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;

int main()
{
    const string model_path = "../models/nanodet.rknn";
    NanoDet net(model_path);
    net.Input_Output_Configuration();
    cv::Mat img = cv::imread("../a.jpeg");

    int height = img.rows, width = img.cols;

    TestTimeVar;
    TestTimeTic;
    auto res = net.detect(img);
    TestTimeTocFPS("forward");

    for (auto some_class_bbxs : res)
    {
        for (auto bbx : some_class_bbxs)
        {
            cv::Rect bbx_(bbx.x1, bbx.y1, bbx.x2 - bbx.x1, bbx.y2 - bbx.y1);
            cv::rectangle(img, bbx_, {0, 0, 225});
            char text[256];
            sprintf(text, "%s %.1f%%", net.labels_[bbx.label].c_str(), bbx.score * 100);
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            int x = (bbx.x1);
            int y = (bbx.y1) - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > img.cols)
                x = img.cols - label_size.width;
            cv::putText(img, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }
    }
    cv::imshow("res", img);
    cv::imwrite("res.jpg", img);
    cv::waitKey(0);
}