#include "meshgrid.h"
#include <Eigen/Eigen>
#include <Eigen/SPQRSupport>
#include<Eigen/SparseCholesky>
#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <armadillo>
#include <fstream>
using namespace std;

#ifndef BUNDLEPATH_H
#define BUNDLEPATH_H
class BundleVideoStab{
    private:
    int codec = CV_FOURCC('D', 'I', 'V', 'X');
    int maxIter = 20;
    int max_corners = 600;   
    int minDistance = 5;
    int span = 30;   //half of winsize
    double quality_level = 0.05;
    double weightGrid = 200;
    double meshSize = 16.0;
    double sigmaT = 10;
    double sigmaM = 800;
    double lambda = 5;
    double stableW = 1;    //weightGrid*20
    string videopath;
    string savevideo;

    void calWeight(vector<vector<cv::Mat> > &omega, vector<cv::Mat> &gamma);
    void getCoherenceTerm(cv::Mat& sumPj,int iter,int t,int row,int col);
    double gauss(double sigma, double x);
    void warping(cv::Mat &Xprime,cv::Mat &Yprime,int t);
    void quadWarp(cv::Mat &Xprime, cv::Mat &Yprime, const Quad &q1, const Quad &q2);
    int min(int a,int b);
    int max(int a,int b);
    public:
        int maxFrames;
        int ImgWidth ;
        int ImgHeight;
        double quadWidth;
        double quadHeight;
        vector<vector<vector<cv::Mat>>> OptimzePath; //iter-time-row-col-H
        vector<vector<cv::Mat>> pathP;
        vector<vector<cv::Mat>> pathC;
        vector<cv::Mat> omegaW;

        BundleVideoStab(string src_path,string dst_path);
        ~BundleVideoStab();
        void MotionEsitmation();        
        void optPath();
        void renderImage();
};



#endif // BUNDLEPATH_H
