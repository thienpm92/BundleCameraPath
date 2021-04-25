#include "bundlepath.h"


BundleVideoStab::BundleVideoStab(string src_path,string dst_path)
{
    videopath = src_path;
    savevideo = dst_path;

    cv::VideoCapture vid(videopath);
    cv::Mat Frame;
    vid>>Frame;
    maxFrames = vid.get(CV_CAP_PROP_FRAME_COUNT);
    cout<<"max frame "<<maxFrames<<endl;
    ImgWidth = Frame.cols ;
    ImgHeight = Frame.rows;
    quadWidth = ImgWidth/meshSize;
    quadHeight = ImgHeight/meshSize;
}

BundleVideoStab::~BundleVideoStab()
{

}


int BundleVideoStab::max(int a,int b){
    return(a>b ? a:b);
}

int BundleVideoStab::min(int a, int b){
    return(a<b ? a:b);
}
