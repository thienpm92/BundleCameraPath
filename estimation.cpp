#include "bundlepath.h"
#include "meshgrid.h"

void BundleVideoStab::MotionEsitmation()
{
    cout<<"MOTION ESTIMATION"<<endl;
    cv::Mat prevFrame;
    cv::Mat curFrame;
    cv::Mat prevGrayFrame;
    cv::Mat curGrayFrame;
        //time-row-col-H
    int IndexFrame = 1;

    cv::VideoCapture vid(videopath);
    vid>>prevFrame;
    cv::cvtColor(prevFrame, prevGrayFrame, CV_BGR2GRAY);

    vector<cv::Mat> initH;
    for(int i=0;i<meshSize;i++){
        for(int j=0;j<meshSize;j++){
            cv::Mat H  = cv::Mat::eye(3,3,CV_64F);
            initH.push_back(H.clone());
        }
    }
    pathC.push_back(initH);
    while(true){
        vid >> curFrame;
        if(curFrame.data==NULL)
            break;
        cout<<IndexFrame<<endl;

        vector<cv::Point2f> pp1,prevFt;
        vector<cv::Point2f> pp2,curFt;
        vector<uchar> status;
        vector<float> err;
        cv::cvtColor(curFrame, curGrayFrame, CV_BGR2GRAY);
        cv::goodFeaturesToTrack(prevGrayFrame, pp1, max_corners, quality_level, minDistance);
        cv::calcOpticalFlowPyrLK(prevGrayFrame, curGrayFrame, pp1, pp2,status , err);

        //Outlier rejection using global H and RANSAC
        int ptCount = status.size();
        cv::Mat p1(ptCount, 2, CV_32F);
        cv::Mat p2(ptCount, 2, CV_32F);
        for (int j = 0; j < ptCount; j++)
        {
            p1.at<float>(j, 0) = pp1[j].x;
            p1.at<float>(j, 1) = pp1[j].y;
            p2.at<float>(j, 0) = pp2[j].x;
            p2.at<float>(j, 1) = pp2[j].y;
        }
        cv::Mat m_Fundamental;
        vector<uchar> m_RANSACStatus;
        m_Fundamental = cv::findFundamentalMat(p1, p2, m_RANSACStatus, cv::FM_RANSAC);
        for (int j = 0; j < ptCount; j++)
        {
            if (m_RANSACStatus[j] == 0)
                status[j] = 0;
        }
        for (int j = 0; j < pp2.size(); j++)
        {
            if (status[j] == 0 || (pp2[j].x <= 0 || pp2[j].y <= 0 || pp2[j].x >= ImgWidth - 1 || pp2[j].y >= ImgHeight - 1))
                continue;
            curFt.push_back(pp2[j]);
            prevFt.push_back(pp1[j]);
        }

        //Mapping Ft to vertexes

        vector<cv::Mat> HomosVec, HomosUpdate;
        MeshGrid AsRigidAsPos(ImgWidth, ImgHeight, quadWidth, quadHeight, weightGrid);
        AsRigidAsPos.SetControlPts(prevFt,curFt);
        AsRigidAsPos.Solve();
        AsRigidAsPos.CalcHomos(HomosVec);


//        if(IndexFrame==0){
//            pathC.push_back(HomosVec);
//            curGrayFrame.copyTo(prevGrayFrame);
//            curFrame.copyTo(prevFrame);
//            IndexFrame++;
//            continue;
//        }
        for(int i=0;i<meshSize*meshSize;i++){
            cv::Mat H = pathC[IndexFrame-1][i]*HomosVec[i];
            HomosUpdate.push_back(H.clone());
        }
        pathC.push_back(HomosUpdate);
        curGrayFrame.copyTo(prevGrayFrame);
        curFrame.copyTo(prevFrame);
        IndexFrame++;
    }
    maxFrames = pathC.size();
//    ofstream Video_traject("Video_traject.txt");
//    Video_traject << "Index par1 par2" << endl;
//    ofstream Video_smooth("Video_smooth.txt");
//    Video_smooth << "Index par1 par2" << endl;
//    for(int t=0;t<(maxFrames);t++){
//        Video_traject << t << " " ;
//        for(int i=0;i<meshSize*meshSize;i++)
//            Video_traject << pathC[t][i].at<double>(0,2) << " ";
//        Video_traject << endl;
//    }
}
