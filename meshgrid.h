#include "mesh.h"
#include "quad.h"
#include "utility.h"
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


#ifndef __MESHFLOW__
#define __MESHFLOW__
class MeshGrid{

//private:
public:
    int m_imgHeight,m_imgWidth;
    double m_quadWidth,m_quadHeight;
    int m_meshheight,m_meshwidth;

    vector<int> x_index;
    vector<int> y_index;
    vector<int> dataterm_element_i;
    vector<int> dataterm_element_j;
    vector<double> dataterm_element_V00;
    vector<double> dataterm_element_V01;
    vector<double> dataterm_element_V10;
    vector<double> dataterm_element_V11;
    vector<cv::Point2f> dataterm_element_orgPt;
    vector<cv::Point2f> dataterm_element_desPt;

    int num_smooth_cons;
    int num_data_cons;
    int num_feature_pts;
    int Scc = 0;
    int DCc = 0;
    int rowCount;
    int columns;

    cv::Mat DataConstraints;
    cv::Mat SmoothConstraints;
    cv::Mat m_source,m_target;
    cv::Mat m_globalHomography;
    Mesh m_mesh;
    Mesh m_warpedmesh;
    vector<cv::Point2f> m_vertexMotion;

    void CreateSmoothCons(double weight);
    void getSmoothWeight(vector<double> &smoothWeight, const cv::Point2f &V1, const cv::Point2f &V2, const cv::Point2f &V3);
    void quadWarp( const cv::Mat src, cv::Mat &Xprime, cv::Mat &Yprime, const Quad &q1, const Quad &q2);
    Eigen::VectorXd CreateDataCons();

    void addCoefficient_1(int i, int j, double weight);
    void addCoefficient_2(int i, int j, double weight);
    void addCoefficient_3(int i, int j, double weight);
    void addCoefficient_4(int i, int j, double weight);
    void addCoefficient_5(int i, int j, double weight);
    void addCoefficient_6(int i, int j, double weight);
    void addCoefficient_7(int i, int j, double weight);
    void addCoefficient_8(int i, int j, double weight);
public:
    MeshGrid(int width, int height, double quadWidth, double quadHeight, double weight);
    void SetControlPts(vector<cv::Point2f> &prevPts, vector<cv::Point2f> &curPts);
    void Solve();
    void CalcHomos(vector<cv::Mat> &homos);
    cv::Mat Warping(cv::Mat &srcImg, const vector<cv::Mat> h);

    cv::Mat WarpingNew(const cv::Mat srcImg);
    void getGridIndAndWeight(cv::Point2f pt, vector<cv::Point2f> &ind, vector<double> &w);
};
#endif
