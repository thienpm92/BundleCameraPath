#include "bundlepath.h"

double BundleVideoStab::gauss(double sigma, double x) {
    double expVal = -1 * pow(x, 2) / (2*pow(sigma, 2));
//    double divider = sqrt(2 * M_PI * pow(sigma, 2));
//    return (1 / divider) * exp(expVal);
    return exp(expVal);
}

void BundleVideoStab::calWeight(vector<vector<cv::Mat>>& omega,vector<cv::Mat>& gamma)
{  
    for(int t=0;t<maxFrames;t++){
        vector<cv::Mat> vecWeight;
        cv::Mat gammaMat = cv::Mat::zeros(meshSize, meshSize, CV_64F);
        cv::Mat WeightMat = cv::Mat::zeros(meshSize, meshSize, CV_64F);
        for(int r = t-span;r<=t+span;r++){
            if (r >= 0 && r < maxFrames && r!=t)
            {
                for(int i=0;i<meshSize;i++){
                    for(int j=0;j<meshSize;j++){
                        double Gt = abs(r-t);
                        double tmp1 = pathC[r][i*meshSize +j].at<double>(0,2);
                        double tmp2 = pathC[t][i*meshSize +j].at<double>(0,2);
                        double tmp3 = pathC[r][i*meshSize +j].at<double>(1,2);
                        double tmp4 = pathC[t][i*meshSize +j].at<double>(1,2);
                        double Gm = abs(tmp1-tmp2) + abs(tmp3-tmp4);
                        double wT = gauss(sigmaT, Gt);
                        double wM = gauss(sigmaM, Gm);
                        double val = wT*wM;
                        WeightMat.at<double>(i,j) = val;
                        gammaMat.at<double>(i,j) += val;
                    }
                }
                vecWeight.push_back(WeightMat.clone());
            }
        }

        for(int i=0;i<meshSize;i++){
            for(int j=0;j<meshSize;j++){
                gammaMat.at<double>(i,j) = gammaMat.at<double>(i,j)*2*lambda + 1;
                if((i==0 || i==(meshSize-1)) && ((j==0) ||(j==(meshSize-1))) ){
                    gammaMat.at<double>(i,j) = gammaMat.at<double>(i,j) + 2*stableW * 3;
                }
                else if((i==0 || i==(meshSize-1)) && ((j>0) ||(j<(meshSize-1))) ){
                    gammaMat.at<double>(i,j) = gammaMat.at<double>(i,j) + 2*stableW * 5;
                }
                else if((i>0 || i<(meshSize-1)) && ((j==0) ||(j==(meshSize-1))) ){
                    gammaMat.at<double>(i,j) = gammaMat.at<double>(i,j) + 2*stableW * 5;
                }
                else if((i>0 || i<(meshSize-1)) && ((j>0) ||(j<(meshSize-1))) ){
                    gammaMat.at<double>(i,j) = gammaMat.at<double>(i,j) + 2*stableW * 8;
                }
            }
        }
        omega.push_back(vecWeight);
        gamma.push_back(gammaMat.clone());
    }
}

void BundleVideoStab::getCoherenceTerm(cv::Mat& sumPj,int iter,int t,int row,int col)
{
    if(row==0 && col==0){   // row=0, col=0
        sumPj += 2*OptimzePath[iter][t][(row+1)*meshSize+col]     * pathC[t][(row+1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*OptimzePath[iter][t][row*meshSize+(col+1)]     * pathC[t][row*meshSize+(col+1)].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*OptimzePath[iter][t][(row+1)*meshSize+(col+1)] * pathC[t][(row+1)*meshSize+(col+1)].inv() * pathC[t][row*meshSize+col];
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col+1)].at<double>(i,j);
//            }
//        }
    }
    else if(row==0 && col >0 && col<(meshSize-1) ){  // row=0, col=1:14
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col]     * pathC[t][(row+1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)]     * pathC[t][row*meshSize+(col+1)].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col+1)] * pathC[t][(row+1)*meshSize+(col+1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)] * pathC[t][(row+1)*meshSize+(col-1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)]     * pathC[t][row*meshSize+(col-1)].inv()     * pathC[t][row*meshSize+col];
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)].at<double>(i,j);
//            }
//        }
    }
    else if(row==0 && col==(meshSize-1)){  //row=0, col=15
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col]     * pathC[t][(row+1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)]     * pathC[t][row*meshSize+(col-1)].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)] * pathC[t][(row+1)*meshSize+(col-1)].inv() * pathC[t][row*meshSize+col];
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)].at<double>(i,j);
//            }
//        }
    }
    else if(row>0 && row<(meshSize-1) && col==0){  //row=1:14, col=0
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col]     * pathC[t][(row+1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col]     * pathC[t][(row-1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col+1)] * pathC[t][(row+1)*meshSize+(col+1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col+1)] * pathC[t][(row-1)*meshSize+(col+1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)]     * pathC[t][row*meshSize+(col+1)].inv()     * pathC[t][row*meshSize+col];
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)].at<double>(i,j);
//            }
//        }
    }
    else if(row>0 && row<(meshSize-1) && col>0 && col<(meshSize-1)){  //row=1:14, col=1:14
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col]     * pathC[t][(row+1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col]     * pathC[t][(row-1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col+1)] * pathC[t][(row+1)*meshSize+(col+1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)] * pathC[t][(row+1)*meshSize+(col-1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col+1)] * pathC[t][(row-1)*meshSize+(col+1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col-1)] * pathC[t][(row-1)*meshSize+(col-1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)]     * pathC[t][row*meshSize+(col+1)].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)]     * pathC[t][row*meshSize+(col-1)].inv()     * pathC[t][row*meshSize+col];
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col-1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)].at<double>(i,j);
//            }
//        }
    }
    else if(row>0 && row<(meshSize-1) && col==(meshSize-1)){  //row=1:14, col=15
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col]     * pathC[t][(row+1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col]     * pathC[t][(row-1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)] * pathC[t][(row+1)*meshSize+(col-1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col-1)] * pathC[t][(row-1)*meshSize+(col-1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)]     * pathC[t][row*meshSize+(col-1)].inv()     * pathC[t][row*meshSize+col];
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row+1)*meshSize+(col-1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col-1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col-1)].at<double>(i,j);

//            }
//        }
    }
    else if(row==(meshSize-1) && col==0){  //row=15, col=0
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col+1)] * pathC[t][(row-1)*meshSize+(col+1)].inv() * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col]     * pathC[t][(row-1)*meshSize+col].inv()     * pathC[t][row*meshSize+col];
        sumPj += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)]     * pathC[t][row*meshSize+(col+1)].inv()     * pathC[t][row*meshSize+col];
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+(col+1)].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][(row-1)*meshSize+col].at<double>(i,j);
//                sumPj.at<double>(i,j) += 2*stableW*OptimzePath[iter][t][row*meshSize+(col+1)].at<double>(i,j);
//            }
//        }
    }
}

void BundleVideoStab::optPath()
{
    cout<<"smoothing"<<endl;
    vector<vector<cv::Mat>> omega;    // [time t][time r][spatial]
    vector<cv::Mat> gamma ;          // [time][spatial-scalar]
    OptimzePath.resize(maxIter);
    OptimzePath[0] = pathC;
    pathP = pathC;          // [time r][spatial][H]

    calWeight(omega,gamma);
    for(int iter=1;iter<maxIter;iter++){
        cout<<"iter "<<iter<<endl;

        //TEMPORAL LOOP
        for(int t=0;t<maxFrames;t++){
            int head = max(t-span,0);
            int tail = min(t+span,maxFrames-1);
            int len = tail-head+1;
            vector<cv::Mat>C = pathC[t];      //[spatial][Mat H]
            vector<cv::Mat>Pj;      //[spatial][Mat H]
            vector<cv::Mat>Pr;      //[spatial][Mat H]
            Pr.resize(meshSize*meshSize);
            Pj.resize(meshSize*meshSize);

            //SPATIAL LOOP
            for(int i=0;i<meshSize;i++){
                for(int j=0;j<meshSize;j++){
                    //second term
                    cv::Mat sumPir = cv::Mat::zeros(3, 3, CV_64F);
                    int idx = 0;
                    for(int r = head;r<=tail;r++){
                        if(r!=t){
                            cv::Mat Pir;
                            OptimzePath[iter-1][r][i*meshSize+j].copyTo(Pir);
                            Pir.at<double>(0,0) = Pir.at<double>(0,0) * omega[t][idx].at<double>(i,j) * 2*lambda;
                            Pir.at<double>(0,1) = Pir.at<double>(0,1) * omega[t][idx].at<double>(i,j) * 2*lambda;
                            Pir.at<double>(0,2) = Pir.at<double>(0,2) * omega[t][idx].at<double>(i,j) * 2*lambda;

                            Pir.at<double>(1,0) = Pir.at<double>(1,0) * omega[t][idx].at<double>(i,j) * 2*lambda;
                            Pir.at<double>(1,1) = Pir.at<double>(1,1) * omega[t][idx].at<double>(i,j) * 2*lambda;
                            Pir.at<double>(1,2) = Pir.at<double>(1,2) * omega[t][idx].at<double>(i,j) * 2*lambda;

                            Pir.at<double>(2,0) = Pir.at<double>(2,0) * omega[t][idx].at<double>(i,j) * 2*lambda;
                            Pir.at<double>(2,1) = Pir.at<double>(2,1) * omega[t][idx].at<double>(i,j) * 2*lambda;
                            Pir.at<double>(2,2) = Pir.at<double>(2,2) * omega[t][idx].at<double>(i,j) * 2*lambda;

                            sumPir += Pir.clone();
                            idx++;
                        }
                    }
                    Pr[i*meshSize+j] = sumPir.clone();
                    //third term
                    cv::Mat sumPj = cv::Mat::zeros(3, 3, CV_64F);
                    getCoherenceTerm(sumPj,iter-1,t,i,j);
                    Pj[i*meshSize+j] = sumPj.clone();
                }
            }
            //UPDATE
            vector<cv::Mat> result;
            for(int i=0;i<meshSize;i++){
                for(int j=0;j<meshSize;j++){
                    cv::Mat updateH = cv::Mat::zeros(3, 3, CV_64F);
                    updateH = (C[i*meshSize+j]+Pr[i*meshSize+j]+Pj[i*meshSize+j])/gamma[t].at<double>(i,j);
                    result.push_back(updateH.clone());
                }
            }            pathP[t] = result;
        }
        OptimzePath[iter] = pathP;
    }
    cout<<"smooth done"<<endl;

    ofstream Video_traject("Video_traject.txt");
    Video_traject << "Index par1 par2 par3 par4" << endl;
    ofstream Video_smooth("Video_smooth.txt");
    Video_smooth << "Index par1 par2" << endl;
    for(int t=0;t<(maxFrames);t++){
        Video_traject << t << " " << pathC[t][0].at<double>(0,2) << " " << pathC[t][0].at<double>(1,2) << " " << pathC[t][0].at<double>(0,0)<< " " << pathC[t][0].at<double>(0,1)<< endl;
        Video_smooth << t << " " << OptimzePath[maxIter-1][t][0].at<double>(0,2) << " " << OptimzePath[maxIter-1][t][0].at<double>(1,2) <<" "<< OptimzePath[maxIter-1][t][0].at<double>(0,0)<<" "<< OptimzePath[maxIter-1][t][0].at<double>(0,1)<< endl;
    }
}
