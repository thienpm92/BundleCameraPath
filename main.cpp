#include "bundlepath.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>
#include "iostream"
#include "fstream"
#include "meshgrid.h"

using namespace std;



int main()
{
    string linkVideo = "/media/eric/DATA/SSU research/Data set/video_test_case/running_4.avi";
    string savevideo = "bundle_running_4.avi";
//    ofstream Video_traject("Video_traject.txt");
//    cv::Mat H  = cv::Mat::eye(3,3,CV_64F);
//    cv::Mat T  = cv::Mat::eye(3,3,CV_64F);
//    H.at<double>(0,0) = 1;
//    H.at<double>(0,1) = 2;
//    H.at<double>(0,2) = 3;
//    H.at<double>(1,0) = 4;
//    H.at<double>(1,1) = 5;
//    H.at<double>(1,2) = 6;

//    cv::Mat B = H.inv();
//    cout<<H<<endl<<endl<<B<<endl<<endl;
//    cv::Mat C = T/H;
//    cout<<C<<endl;

    int start_s=clock();
    BundleVideoStab video(linkVideo,savevideo);
    video.MotionEsitmation();
    video.optPath();
    video.renderImage();
    cout<<"done"<<endl;
    int stop_s=clock();
    cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC) << endl;
    return 0;
}
