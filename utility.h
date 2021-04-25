#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <armadillo>
#include <fstream>

using namespace std;

#ifndef UTILITY_H
#define UTILITY_H
template<typename T, int N>
vector<double> bilinear(const T *const data, const int w, const int h, const cv::Point2f &loc) {
    const double epsilon = 0.00001;
    int xl = floor(loc.x - epsilon), xh = (int) round(loc.x + 0.5 - epsilon);
    int yl = floor(loc.y - epsilon), yh = (int) round(loc.y + 0.5 - epsilon);

    if (loc.x <= epsilon)
        xl = 0;
    if (loc.y <= epsilon)
        yl = 0;

    const int l1 = yl * w + xl;
    const int l2 = yh * w + xh;
    if (l1 == l2) {
        vector<double> res(N);
        for (size_t i = 0; i < N; ++i)
            res[i] = data[l1 * N + i];
        return res;
    }

    double lm = loc.x - (double) xl, rm = (double) xh - loc.x;
    double tm = loc.y - (double) yl, bm = (double) yh - loc.y;
    vector<int> ind(4);
    ind[0] = xl + yl * w;
    ind[1] = xh + yl * w;
    ind[2] = xh + yh * w;
    ind[3] = xl + yh * w;

    std::vector<vector<double>> v(4);
    for (size_t i = 0; i < 4; i++)
        v[i].resize(3,0);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < N; ++j)
            v[i][j] = data[ind[i] * N + j];
    }
    if (std::abs(lm) <= epsilon && std::abs(rm) <= epsilon){
        vector<double> res(N);
        res[0] = (v[0][0] * bm + v[2][0] * tm) / (bm + tm);
        res[1] = (v[0][1] * bm + v[2][1] * tm) / (bm + tm);
        res[2] = (v[0][2] * bm + v[2][2] * tm) / (bm + tm);
        return res;
    }

    if (std::abs(bm) <= epsilon && std::abs(tm) <= epsilon){
        vector<double> res(N);
        res[0] = (v[0][0] * rm + v[2][0] * lm) / (lm + rm);
        res[1] = (v[0][1] * rm + v[2][1] * lm) / (lm + rm);
        res[2] = (v[0][2] * rm + v[2][2] * lm) / (lm + rm);
        return res;
    }

    vector<double> vw(4);
    vw[0] = rm * bm;
    vw[1] = lm * bm;
    vw[2] = lm * tm;
    vw[3] = rm * tm;
    double sum = vw[0]+vw[1]+vw[2]+vw[3];
    vector<double> res(N);
    res[0] = (v[0][0] * vw[0] + v[1][0] * vw[1] + v[2][0] * vw[2] + v[3][0] * vw[3]) / sum;
    res[1] = (v[0][1] * vw[0] + v[1][1] * vw[1] + v[2][1] * vw[2] + v[3][1] * vw[3]) / sum;
    res[2] = (v[0][2] * vw[0] + v[1][2] * vw[1] + v[2][2] * vw[2] + v[3][2] * vw[3]) / sum;

    return res;
};
#endif // UTILITY_H
