#include "bundlepath.h"

void BundleVideoStab::renderImage()
{
    cout<<"rendering"<<endl;
    cv::VideoCapture video(videopath);
    cv::VideoWriter record(savevideo, codec, 30, cvSize(ImgWidth, ImgHeight), true);
    cv::Mat orgFrame;

    pathP = OptimzePath[maxIter-1];
    for(int t=0;t<maxFrames;t++){
        video>>orgFrame;
        cv::Mat wrpFrame = cv::Mat::zeros( orgFrame.size(), orgFrame.type() );
        cv::Mat Xprime = cv::Mat::zeros( orgFrame.size(), CV_32FC1 );
        cv::Mat Yprime = cv::Mat::zeros( orgFrame.size(), CV_32FC1 );

        warping(Xprime,Yprime, t);
        cv::Vec3b color;
        color[0] = 0;
        color[1] = 0;
        color[2] = 0;
        orgFrame.at<cv::Vec3b>(cv::Point(0,0)) = color;
        cv::remap(orgFrame, wrpFrame, Xprime, Yprime, CV_INTER_CUBIC, cv::BORDER_TRANSPARENT);
        record.write(wrpFrame);
    }
    cout<<"render done"<<endl;
}

void BundleVideoStab::warping(cv::Mat &Xprime,cv::Mat &Yprime,int t)
{
    Mesh srcMesh(ImgHeight,ImgWidth,quadWidth,quadHeight);
    Mesh dstMesh(ImgHeight,ImgWidth,quadWidth,quadHeight);

    for(int i=0;i<=meshSize;i++){
        for(int j=0;j<=meshSize;j++){
            cv::Point2f node = srcMesh.getVertex(i,j);
            double x = node.x;
            double y = node.y;
            if(i==0 && j==0)    //cell left-top
            {
                cv::Mat C = pathC[t][i*meshSize+j];
                cv::Mat P = pathP[t][i*meshSize+j];
//                cv::Mat B = P*C.inv();
                cv::Mat B = C.inv()*P;

                double X = B.at<double>(0, 0) * x + B.at<double>(0, 1) * y + B.at<double>(0, 2);
                double Y = B.at<double>(1, 0) * x + B.at<double>(1, 1) * y + B.at<double>(1, 2);
                double W = B.at<double>(2, 0) * x + B.at<double>(2, 1) * y + B.at<double>(2, 2);
                X = X/W;
                Y = Y/W;
                cv::Point2f pt(X,Y);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i==0 && j==meshSize)    //cell right-top
            {
                cv::Mat C = pathC[t][i*meshSize+j-1];
                cv::Mat P = pathP[t][i*meshSize+j-1];
//                cv::Mat B = P*C.inv();
                cv::Mat B = C.inv()*P;

                double X = B.at<double>(0, 0) * x + B.at<double>(0, 1) * y + B.at<double>(0, 2);
                double Y = B.at<double>(1, 0) * x + B.at<double>(1, 1) * y + B.at<double>(1, 2);
                double W = B.at<double>(2, 0) * x + B.at<double>(2, 1) * y + B.at<double>(2, 2);
                X = X/W;
                Y = Y/W;
                cv::Point2f pt(X,Y);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i==0 && j>0 && j<meshSize)    //cell top
            {
                cv::Mat C1 = pathC[t][i*meshSize+j-1];
                cv::Mat P1 = pathP[t][i*meshSize+j-1];
//                cv::Mat B1 = P1*C1.inv();
                cv::Mat B1 = C1.inv()*P1;
                cv::Mat C2 = pathC[t][i*meshSize+j];
                cv::Mat P2 = pathP[t][i*meshSize+j];
//                cv::Mat B2 = P2*C2.inv();
                cv::Mat B2 = C2.inv()*P2;

                double X1 = B1.at<double>(0, 0) * x + B1.at<double>(0, 1) * y + B1.at<double>(0, 2);
                double Y1 = B1.at<double>(1, 0) * x + B1.at<double>(1, 1) * y + B1.at<double>(1, 2);
                double W1 = B1.at<double>(2, 0) * x + B1.at<double>(2, 1) * y + B1.at<double>(2, 2);
                X1 = X1/W1;
                Y1 = Y1/W1;
                double X2 = B2.at<double>(0, 0) * x + B2.at<double>(0, 1) * y + B2.at<double>(0, 2);
                double Y2 = B2.at<double>(1, 0) * x + B2.at<double>(1, 1) * y + B2.at<double>(1, 2);
                double W2 = B2.at<double>(2, 0) * x + B2.at<double>(2, 1) * y + B2.at<double>(2, 2);
                X2 = X2/W2;
                Y2 = Y2/W2;
                double meanX = (X1+X2)/2;
                double meanY = (Y1+Y2)/2;
                cv::Point2f pt(meanX,meanY);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i>0 && i<meshSize && j==0)    //cell left
            {
                cv::Mat C1 = pathC[t][i*meshSize+j];
                cv::Mat P1 = pathP[t][i*meshSize+j];
//                cv::Mat B1 = P1*C1.inv();
                cv::Mat B1 = C1.inv()*P1;
                cv::Mat C2 = pathC[t][(i-1)*meshSize+j];
                cv::Mat P2 = pathP[t][(i-1)*meshSize+j];
//                cv::Mat B2 = P2*C2.inv();
                cv::Mat B2 = C2.inv()*P2;

                double X1 = B1.at<double>(0, 0) * x + B1.at<double>(0, 1) * y + B1.at<double>(0, 2);
                double Y1 = B1.at<double>(1, 0) * x + B1.at<double>(1, 1) * y + B1.at<double>(1, 2);
                double W1 = B1.at<double>(2, 0) * x + B1.at<double>(2, 1) * y + B1.at<double>(2, 2);
                X1 = X1/W1;
                Y1 = Y1/W1;
                double X2 = B2.at<double>(0, 0) * x + B2.at<double>(0, 1) * y + B2.at<double>(0, 2);
                double Y2 = B2.at<double>(1, 0) * x + B2.at<double>(1, 1) * y + B2.at<double>(1, 2);
                double W2 = B2.at<double>(2, 0) * x + B2.at<double>(2, 1) * y + B2.at<double>(2, 2);
                X2 = X2/W2;
                Y2 = Y2/W2;
                double meanX = (X1+X2)/2;
                double meanY = (Y1+Y2)/2;
                cv::Point2f pt(meanX,meanY);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i>0 && i<meshSize && j==meshSize)    //cell right
            {
                cv::Mat C1 = pathC[t][i*meshSize+j-1];
                cv::Mat P1 = pathP[t][i*meshSize+j-1];
//                cv::Mat B1 = P1*C1.inv();
                cv::Mat B1 = C1.inv()*P1;
                cv::Mat C2 = pathC[t][(i-1)*meshSize+j-1];
                cv::Mat P2 = pathP[t][(i-1)*meshSize+j-1];
//                cv::Mat B2 = P2*C2.inv();
                cv::Mat B2 = C2.inv()*P2;

                double X1 = B1.at<double>(0, 0) * x + B1.at<double>(0, 1) * y + B1.at<double>(0, 2);
                double Y1 = B1.at<double>(1, 0) * x + B1.at<double>(1, 1) * y + B1.at<double>(1, 2);
                double W1 = B1.at<double>(2, 0) * x + B1.at<double>(2, 1) * y + B1.at<double>(2, 2);
                X1 = X1/W1;
                Y1 = Y1/W1;
                double X2 = B2.at<double>(0, 0) * x + B2.at<double>(0, 1) * y + B2.at<double>(0, 2);
                double Y2 = B2.at<double>(1, 0) * x + B2.at<double>(1, 1) * y + B2.at<double>(1, 2);
                double W2 = B2.at<double>(2, 0) * x + B2.at<double>(2, 1) * y + B2.at<double>(2, 2);
                X2 = X2/W2;
                Y2 = Y2/W2;
                double meanX = (X1+X2)/2;
                double meanY = (Y1+Y2)/2;
                cv::Point2f pt(meanX,meanY);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i==meshSize && j==0)    //cell left-bot
            {
                cv::Mat C = pathC[t][(i-1)*meshSize+j];
                cv::Mat P = pathP[t][(i-1)*meshSize+j];
//                cv::Mat B = P*C.inv();
                cv::Mat B = C.inv()*P;

                double X = B.at<double>(0, 0) * x + B.at<double>(0, 1) * y + B.at<double>(0, 2);
                double Y = B.at<double>(1, 0) * x + B.at<double>(1, 1) * y + B.at<double>(1, 2);
                double W = B.at<double>(2, 0) * x + B.at<double>(2, 1) * y + B.at<double>(2, 2);
                X = X/W;
                Y = Y/W;
                cv::Point2f pt(X,Y);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i==meshSize && j==meshSize)    //cell right-bot
            {
                cv::Mat C = pathC[t][(i-1)*meshSize+j-1];
                cv::Mat P = pathP[t][(i-1)*meshSize+j-1];
//                cv::Mat B = P*C.inv();
                cv::Mat B = C.inv()*P;

                double X = B.at<double>(0, 0) * x + B.at<double>(0, 1) * y + B.at<double>(0, 2);
                double Y = B.at<double>(1, 0) * x + B.at<double>(1, 1) * y + B.at<double>(1, 2);
                double W = B.at<double>(2, 0) * x + B.at<double>(2, 1) * y + B.at<double>(2, 2);
                X = X/W;
                Y = Y/W;
                cv::Point2f pt(X,Y);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i==meshSize && j>0 && j<meshSize)    //cell bot
            {
                cv::Mat C1 = pathC[t][(i-1)*meshSize+j-1];
                cv::Mat P1 = pathP[t][(i-1)*meshSize+j-1];
//                cv::Mat B1 = P1*C1.inv();
                cv::Mat B1 = C1.inv()*P1;
                cv::Mat C2 = pathC[t][(i-1)*meshSize+j];
                cv::Mat P2 = pathP[t][(i-1)*meshSize+j];
//                cv::Mat B2 = P2*C2.inv();
                cv::Mat B2 = C2.inv()*P2;

                double X1 = B1.at<double>(0, 0) * x + B1.at<double>(0, 1) * y + B1.at<double>(0, 2);
                double Y1 = B1.at<double>(1, 0) * x + B1.at<double>(1, 1) * y + B1.at<double>(1, 2);
                double W1 = B1.at<double>(2, 0) * x + B1.at<double>(2, 1) * y + B1.at<double>(2, 2);
                X1 = X1/W1;
                Y1 = Y1/W1;
                double X2 = B2.at<double>(0, 0) * x + B2.at<double>(0, 1) * y + B2.at<double>(0, 2);
                double Y2 = B2.at<double>(1, 0) * x + B2.at<double>(1, 1) * y + B2.at<double>(1, 2);
                double W2 = B2.at<double>(2, 0) * x + B2.at<double>(2, 1) * y + B2.at<double>(2, 2);
                X2 = X2/W2;
                Y2 = Y2/W2;
                double meanX = (X1+X2)/2;
                double meanY = (Y1+Y2)/2;
                cv::Point2f pt(meanX,meanY);
                dstMesh.setVertex(i,j,pt);
            }
            else if(i>0 && i<meshSize && j>0 && j<meshSize)    //cell central
            {
                cv::Mat C1 = pathC[t][(i-1)*meshSize+j-1];
                cv::Mat P1 = pathP[t][(i-1)*meshSize+j-1];
//                cv::Mat B1 = P1*C1.inv();
                cv::Mat B1 = C1.inv()*P1;
                cv::Mat C2 = pathC[t][(i-1)*meshSize+j];
                cv::Mat P2 = pathP[t][(i-1)*meshSize+j];
//                cv::Mat B2 = P2*C2.inv();
                cv::Mat B2 = C2.inv()*P2;
                cv::Mat C3 = pathC[t][i*meshSize+j-1];
                cv::Mat P3 = pathP[t][i*meshSize+j-1];
//                cv::Mat B3 = P3*C3.inv();
                cv::Mat B3 = C3.inv()*P3;
                cv::Mat C4 = pathC[t][i*meshSize+j];
                cv::Mat P4 = pathP[t][i*meshSize+j];
//                cv::Mat B4 = P4*C4.inv();
                cv::Mat B4 = C4.inv()*P4;

                double X1 = B1.at<double>(0, 0) * x + B1.at<double>(0, 1) * y + B1.at<double>(0, 2);
                double Y1 = B1.at<double>(1, 0) * x + B1.at<double>(1, 1) * y + B1.at<double>(1, 2);
                double W1 = B1.at<double>(2, 0) * x + B1.at<double>(2, 1) * y + B1.at<double>(2, 2);
                X1 = X1/W1;
                Y1 = Y1/W1;
                double X2 = B2.at<double>(0, 0) * x + B2.at<double>(0, 1) * y + B2.at<double>(0, 2);
                double Y2 = B2.at<double>(1, 0) * x + B2.at<double>(1, 1) * y + B2.at<double>(1, 2);
                double W2 = B2.at<double>(2, 0) * x + B2.at<double>(2, 1) * y + B2.at<double>(2, 2);
                X2 = X2/W2;
                Y2 = Y2/W2;
                double X3 = B3.at<double>(0, 0) * x + B3.at<double>(0, 1) * y + B3.at<double>(0, 2);
                double Y3 = B3.at<double>(1, 0) * x + B3.at<double>(1, 1) * y + B3.at<double>(1, 2);
                double W3 = B3.at<double>(2, 0) * x + B3.at<double>(2, 1) * y + B3.at<double>(2, 2);
                X3 = X3/W3;
                Y3 = Y3/W3;
                double X4 = B4.at<double>(0, 0) * x + B4.at<double>(0, 1) * y + B4.at<double>(0, 2);
                double Y4 = B4.at<double>(1, 0) * x + B4.at<double>(1, 1) * y + B4.at<double>(1, 2);
                double W4 = B4.at<double>(2, 0) * x + B4.at<double>(2, 1) * y + B4.at<double>(2, 2);
                X4 = X4/W4;
                Y4 = Y4/W4;
                double meanX = (X1+X2+X3+X4)/4;
                double meanY = (Y1+Y2+Y3+Y4)/4;
                cv::Point2f pt(meanX,meanY);
                dstMesh.setVertex(i,j,pt);
            }
        }
    }

    for (int i = 1; i <= meshSize; i ++){
        for (int j = 1; j <= meshSize; j ++){
            cv::Point2f p0 = srcMesh.getVertex(i-1,j-1);
            cv::Point2f p1 = srcMesh.getVertex(i-1,j);
            cv::Point2f p2 = srcMesh.getVertex(i,j-1);
            cv::Point2f p3 = srcMesh.getVertex(i,j);

            cv::Point2f q0 = dstMesh.getVertex(i-1,j-1);
            cv::Point2f q1 = dstMesh.getVertex(i-1,j);
            cv::Point2f q2 = dstMesh.getVertex(i,j-1);
            cv::Point2f q3 = dstMesh.getVertex(i,j);

            Quad quad1(p0, p1, p2, p3);
            Quad quad2(q0, q1, q2, q3);

            quadWarp(Xprime, Yprime,quad1, quad2);
        }
    }
}

void BundleVideoStab::quadWarp(cv::Mat &Xprime, cv::Mat &Yprime, const Quad &q1, const Quad &q2){
    int minx = max(0, (int)floor(q2.getMinX()));
    int maxx = min(ImgWidth-1, (int)ceil(q2.getMaxX()));
    int miny = max(0, (int)floor(q2.getMinY()));
    int maxy = min(ImgHeight-1, (int)ceil(q2.getMaxY()));
    int tmpx = maxx;
    int tmpy = maxy;
    if(minx!=0)
        tmpx = maxx+1;
    if(miny!=0)
        tmpy = maxy+1;


    vector<cv::Point2f> source(4);
    vector<cv::Point2f> target(4);
    source[0] = q2.V00;
    source[1] = q2.V01;
    source[2] = q2.V10;
    source[3] = q2.V11;

    target[0] = q1.V00;
    target[1] = q1.V01;
    target[2] = q1.V10;
    target[3] = q1.V11;
    cv::Mat H = cv::findHomography(source, target, 0);


    for (int i = miny; i < tmpy; i ++) {
        for (int j = minx; j < tmpx; j ++) {
            double X = H.at<double>(0, 0) * j + H.at<double>(0, 1) * i + H.at<double>(0, 2);
            double Y = H.at<double>(1, 0) * j + H.at<double>(1, 1) * i + H.at<double>(1, 2);
            double W = H.at<double>(2, 0) * j + H.at<double>(2, 1) * i + H.at<double>(2, 2);
            W = W ? 1.0 / W : 0;
            X = X*W;
            Y = Y*W;
            Xprime.at<float>(i, j) = X;
            Yprime.at<float>(i, j) = Y;
        }
    }
}
