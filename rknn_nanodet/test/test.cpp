#include <unistd.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <string>
#include <vector>

#include "NanoDet.hpp"
#include "testtime.hpp"
using namespace std;

int main(int argc, const char** argv) {
    const string model_path = "/home/toybrick/osnet_pedestrain_detection/models/nanodet.rknn";
    const string seq_path = "/home/toybrick/VisDrone/SOT/VisDrone2018-SOT-test-dev/sequences/uav0000021_00000_s";

    vector<string> all_imgs;
    cv::glob(seq_path + "/*.jpg", all_imgs);
    // create net

    vector<cv::Ptr<cv::TrackerKCF>> trackers;
    while (1) {
        trackers.clear();
        cv::Mat img = cv::imread(all_imgs[0]);
        vector<cv::Ptr<cv::TrackerKCF>> newtrackers;
        for (int i = 0; i < 1000; ++i) {
            cv::TrackerKCF::Params paras;
            paras.detect_thresh = 0.35;
            cv::Ptr<cv::TrackerKCF> pTck = cv::TrackerKCF::create(paras);
            pTck->init(img, cv::Rect(30, 30, 50, 60));
            newtrackers.push_back(pTck);
        }
        usleep(1000000);
        trackers = newtrackers;
    }

    NanoDet net(model_path, "");

    int nframe = all_imgs.size();
    TestTimeVar;
    for (int i = 0; i < all_imgs.size(); ++i) {
        cv::Mat img = cv::imread(all_imgs[i]);

        int imgw = img.cols, imgh = img.rows;
        cout << imgw << " " << imgh << endl;
        std::vector<cv::Rect> allbbx;
        TestTimeTic;
        for (int i = 0; i < imgw / 320 - 1; i++) {
            for (int j = 0; j < imgh / 320 - 1; j++) {
                cv::Mat patch = img(cv::Rect(cv::Point(i * 320, j * 320), cv::Point((i + 1) * 320, (j + 1) * 320)));
                std::vector<cv::Rect> bbx = net.detect(patch);
                cv::imshow("patch", patch);
                cout << bbx.size() << endl;
                for (auto _bbx : bbx) {
                    _bbx += cv::Point2i(i * 320, j * 320);
                    allbbx.push_back(_bbx);
                }
                cv::waitKey(10);
            }
        }

        TestTimeTocFPS("run : ");
        cout << allbbx.size() << endl;
        for (auto _bbx : allbbx) {
            cv::rectangle(img, _bbx, {0, 0, 225});
            // cout << _bbx << endl;
        }
        // cv::imwrite("a.jpeg", img);
        cv::imshow("dsaf", img);
        cv::waitKey(10);
    }
    return 0;
}
