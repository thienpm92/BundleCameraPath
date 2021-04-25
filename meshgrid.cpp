#include "meshgrid.h"
#include "utility.h"

MeshGrid::MeshGrid(int width, int height, double quadWidth, double quadHeight, double weight){

    m_imgHeight  = height;
    m_imgWidth   = width;
    m_quadWidth  = quadWidth;
    m_quadHeight = quadHeight;


//    m_mesh       = new Mesh(height,width,quadWidth,quadHeight);
//    m_warpedmesh = new Mesh(height,width,quadWidth,quadHeight);
    Mesh initMesh(height,width,quadWidth,quadHeight);
    m_mesh = initMesh;
    m_warpedmesh = initMesh;
    m_meshheight = m_mesh.meshHeight;
    m_meshwidth  = m_mesh.meshWidth;
    m_vertexMotion.resize(m_meshheight*m_meshwidth);


//    x_index.resize(m_meshheight*m_meshwidth);
//    y_index.resize(m_meshheight*m_meshwidth);
    for(int i=0;i<m_meshheight*m_meshwidth;i++){
        x_index.push_back(i);
        y_index.push_back(m_meshheight*m_meshwidth+i);
    }
    columns = m_meshheight*m_meshwidth*2;
    num_smooth_cons = (m_meshheight-2)*(m_meshwidth-2)*16 + ((m_meshwidth+m_meshheight)*2-8)*8 + 16;
    SmoothConstraints = cv::Mat::zeros(num_smooth_cons*5, 3, CV_64FC1);     //size calculate base on number of weight vertice in the code
    CreateSmoothCons(weight);
}

void MeshGrid::SetControlPts(vector<cv::Point2f> &prevPts, vector<cv::Point2f> &curPts){
    int len = prevPts.size();
    dataterm_element_i.resize(len,0);
    dataterm_element_j.resize(len,0);
    dataterm_element_V00.resize(len,0);
    dataterm_element_V01.resize(len,0);
    dataterm_element_V10.resize(len,0);
    dataterm_element_V11.resize(len,0);
    num_feature_pts = prevPts.size();

    dataterm_element_orgPt = prevPts;
    dataterm_element_desPt = curPts;

    for(int i=0;i<len;i++){
        cv::Point2f pt = prevPts[i];
        dataterm_element_i[i] = floor(pt.y/m_quadHeight)+ 1;                                            //????????
        dataterm_element_j[i] = floor(pt.x/m_quadWidth) + 1;
        Quad quad = m_mesh.getQuad(dataterm_element_i[i],dataterm_element_j[i]);
        vector<double> coefficient;
        quad.getBilinearCoordinates(pt, coefficient );
        dataterm_element_V00[i] = coefficient[0];
        dataterm_element_V01[i] = coefficient[1];
        dataterm_element_V10[i] = coefficient[2];
        dataterm_element_V11[i] = coefficient[3];
    }
}

void MeshGrid::Solve(){
    Eigen::VectorXd B = CreateDataCons();
    int B_size = num_data_cons + num_smooth_cons;
    int N = SmoothConstraints.rows + DataConstraints.rows;
    cv::Mat DataMat;
    DataMat= cv::Mat::zeros(N, N, CV_64FC1);
    vector<Eigen::Triplet<double>> triplets;

//optimize computation time so update all constraint and triplet in one loop
    for(int i=0;i<SmoothConstraints.rows;i++){
        int y = SmoothConstraints.at<double>(i,0)+1;
        int x = SmoothConstraints.at<double>(i,1);
        DataMat.at<double>(x,y) += SmoothConstraints.at<double>(i,2);
        double val = DataMat.at<double>(x,y);
        if(val!=0){
            triplets.push_back(Eigen::Triplet<double>(y,x,val));
        }
        if(i<DataConstraints.rows){
            int yi = DataConstraints.at<double>(i,0)+1;
            int xi = DataConstraints.at<double>(i,1);
            DataMat.at<double>(xi,yi) += DataConstraints.at<double>(i,2);
            val = DataMat.at<double>(xi,yi);
            if(val!=0){
                triplets.push_back(Eigen::Triplet<double>(yi,xi,val));
            }
        }
    }

    Eigen::SparseMatrix<double> A(B_size,  m_meshheight*m_meshwidth*2);
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SPQR<Eigen::SparseMatrix<double> > solver(A);
    Eigen::VectorXd X = solver.solve(B.block(0,0,B_size,1));
    int halfcolumns = columns/2;
    for(int i=0;i<m_meshheight;i++){
        for(int j=0;j<m_meshwidth;j++){
            cv::Point2f pt(X[i*m_meshwidth+j],X[halfcolumns+i*m_meshwidth+j]);
            m_warpedmesh.setVertex(i,j,pt);
        }
    }
}

void MeshGrid::CalcHomos(vector<cv::Mat> &homos){
    vector<cv::Point2f> source(4);
    vector<cv::Point2f> target(4);

    for(int i=1;i<m_meshheight;i++){
        for(int j=1;j<m_meshwidth;j++){
            Quad s = m_mesh.getQuad(i, j);
            Quad t = m_warpedmesh.getQuad(i, j);

            source[0] = s.V00;
            source[1] = s.V01;
            source[2] = s.V10;
            source[3] = s.V11;

            target[0] = t.V00;
            target[1] = t.V01;
            target[2] = t.V10;
            target[3] = t.V11;

            cv::Mat H = cv::findHomography(source, target, 0);
            cv::Mat T = estimateRigidTransform(source, target, false);
            homos.push_back(H.clone());
        }
    }
}

cv::Mat MeshGrid::Warping(cv::Mat &srcImg, const vector<cv::Mat> homos){
    cv::Mat warpImg;
    warpImg = cv::Mat::zeros( srcImg.size(), srcImg.type() );
    cv::Mat Xprime = cv::Mat::zeros( srcImg.size(), CV_32FC1 );
    cv::Mat Yprime = cv::Mat::zeros( srcImg.size(), CV_32FC1 );
    for (int i = 1; i < m_meshheight; i ++)	{
        for (int j = 1; j < m_meshwidth; j ++)		{
            cv::Point2f p0 = m_mesh.getVertex(i-1, j-1);
            cv::Point2f p1 = m_mesh.getVertex(i-1, j);
            cv::Point2f p2 = m_mesh.getVertex(i, j-1);
            cv::Point2f p3 = m_mesh.getVertex(i,j);

            cv::Point2f q0 = m_warpedmesh.getVertex(i-1, j-1);
            cv::Point2f q1 = m_warpedmesh.getVertex(i-1, j);
            cv::Point2f q2 = m_warpedmesh.getVertex(i, j-1);
            cv::Point2f q3 = m_warpedmesh.getVertex(i, j);

            Quad quad1(p0, p1, p2, p3);
            Quad quad2(q0, q1, q2, q3);

            quadWarp(srcImg, Xprime, Yprime,quad1, quad2);
        }
    }
    cv::Vec3b color;
    color[0] = 0;
    color[1] = 0;
    color[2] = 0;
    srcImg.at<cv::Vec3b>(cv::Point(0,0)) = color;
    cv::remap(srcImg, warpImg, Xprime, Yprime, CV_INTER_CUBIC, cv::BORDER_TRANSPARENT);
    return warpImg;
}



void MeshGrid::quadWarp( const cv::Mat src, cv::Mat &Xprime, cv::Mat &Yprime, const Quad &q1, const Quad &q2){
    int minx = max(0, (int)floor(q2.getMinX()));
    int maxx = min(src.cols-1, (int)ceil(q2.getMaxX()));
    int miny = max(0, (int)floor(q2.getMinY()));
    int maxy = min(src.rows-1, (int)ceil(q2.getMaxY()));
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

Eigen::VectorXd MeshGrid::CreateDataCons(){
    int len = dataterm_element_i.size();        //number of feature points
    num_data_cons = len*2;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(num_data_cons + num_smooth_cons);
    DataConstraints = cv::Mat::zeros(len*8, 3, CV_64FC1);       //size base on measure

    rowCount--;
    for(int k=0;k<len;k++){
        int i = dataterm_element_i[k];
        int j = dataterm_element_j[k];
        double v00 = dataterm_element_V00[k];
        double v01 = dataterm_element_V01[k];
        double v10 = dataterm_element_V10[k];
        double v11 = dataterm_element_V11[k];

        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = x_index[(i-1)*m_meshwidth+j-1];           //????????
        DataConstraints.at<double>(DCc,2) = v00;
        DCc++;
        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = x_index[(i-1)*m_meshwidth+j];
        DataConstraints.at<double>(DCc,2) = v01;
        DCc++;
        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = x_index[i*m_meshwidth+j-1];
        DataConstraints.at<double>(DCc,2) = v10;
        DCc++;
        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = x_index[i*m_meshwidth+j];
        DataConstraints.at<double>(DCc,2) = v11;
        DCc++;
        rowCount++;

        b[rowCount] = dataterm_element_desPt[k].x;

        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = y_index[(i-1)*m_meshwidth+j-1];
        DataConstraints.at<double>(DCc,2) = v00;
        DCc++;
        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = y_index[(i-1)*m_meshwidth+j];
        DataConstraints.at<double>(DCc,2) = v01;
        DCc++;
        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = y_index[i*m_meshwidth+j-1];
        DataConstraints.at<double>(DCc,2) = v10;
        DCc++;
        DataConstraints.at<double>(DCc,0) = rowCount;
        DataConstraints.at<double>(DCc,1) = y_index[i*m_meshwidth+j];
        DataConstraints.at<double>(DCc,2) = v11;
        DCc++;
        rowCount++;

        b[rowCount] = dataterm_element_desPt[k].y;
    }
    return b;
}

void MeshGrid::CreateSmoothCons(double weight){
    rowCount = 0;
    int i=0;
    int j=0;
    addCoefficient_5(i,j,weight);
    addCoefficient_6(i,j,weight);

    i=0;
    j = m_meshwidth-1;
    addCoefficient_7(i,j,weight);
    addCoefficient_8(i,j,weight);

    i=m_meshheight-1;
    j=0;
    addCoefficient_3(i,j,weight);
    addCoefficient_4(i,j,weight);

    i=m_meshheight-1;
    j=m_meshwidth-1;
    addCoefficient_1(i,j,weight);
    addCoefficient_2(i,j,weight);

    i=0;
    for(j=1;j<m_meshwidth-1;j++){
        addCoefficient_5(i,j,weight);
        addCoefficient_6(i,j,weight);
        addCoefficient_7(i,j,weight);
        addCoefficient_8(i,j,weight);
    }

    i=m_meshheight-1;
    for(j=1;j<m_meshwidth-1;j++){
        addCoefficient_1(i,j,weight);
        addCoefficient_2(i,j,weight);
        addCoefficient_3(i,j,weight);
        addCoefficient_4(i,j,weight);
    }

    j=0;
    for(i=1;i<m_meshheight-1;i++){
        addCoefficient_3(i,j,weight);
        addCoefficient_4(i,j,weight);
        addCoefficient_5(i,j,weight);
        addCoefficient_6(i,j,weight);
    }

    j=m_meshwidth-1;
    for(i=1;i<m_meshheight-1;i++){
        addCoefficient_1(i,j,weight);
        addCoefficient_2(i,j,weight);
        addCoefficient_7(i,j,weight);
        addCoefficient_8(i,j,weight);
    }

    for(i=1;i<m_meshheight-1;i++){
        for(j=1;j<m_meshwidth-1;j++){
            addCoefficient_1(i,j,weight);
            addCoefficient_2(i,j,weight);
            addCoefficient_3(i,j,weight);
            addCoefficient_4(i,j,weight);
            addCoefficient_5(i,j,weight);
            addCoefficient_6(i,j,weight);
            addCoefficient_7(i,j,weight);
            addCoefficient_8(i,j,weight);
        }
    }
}

void MeshGrid::getSmoothWeight(vector<double> &smoothWeight, const cv::Point2f &V1, const cv::Point2f &V2, const cv::Point2f &V3){
    double d1 = sqrt((V1.x - V2.x)*(V1.x - V2.x) + (V1.y - V2.y)*(V1.y - V2.y));
    double d3 = sqrt((V2.x - V3.x)*(V2.x - V3.x) + (V2.y - V3.y)*(V2.y - V3.y));

    cv::Point2f V21(V1.x-V2.x, V1.y-V2.y);
    cv::Point2f V23(V3.x-V2.x, V3.y-V2.y);

    double cosin = V21.x*V23.x + V21.y*V23.y;
    cosin = cosin/(d1*d3);
    double u_dis = cosin*d1;
    double u = u_dis/d3;

    double v_dis = sqrt(d1*d1 - u_dis*u_dis);
    double v = v_dis/d3;
    smoothWeight[0] = u;
    smoothWeight[1] = v;
}






void MeshGrid::addCoefficient_1(int i, int j, double weight){
    //  V3(i-1,j-1)
    //  |
    //  |
    //  |_________V1(i,j)
    //  V2
    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i,j-1);
    cv::Point2f V3 = m_mesh.getVertex(i-1,j-1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*i + j - 1;
    int coordv3 = m_meshwidth*(i-1) + j - 1;

//    coordv1 = coordv1 + 1;                                                              //????????
//    coordv2 = coordv2 + 1;
//    coordv3 = coordv3 + 1;


    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

void MeshGrid::addCoefficient_2(int i, int j, double weight){
    //  V3     V2
    //  _______
    //          |
    //          |
    //          |
    //          V1

    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i-1,j);
    cv::Point2f V3 = m_mesh.getVertex(i-1,j-1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*(i-1) + j;
    int coordv3 = m_meshwidth*(i-1) + j - 1;

    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

void MeshGrid::addCoefficient_3(int i, int j, double weight){
    //  V2     V3
    //  _______
    //  |
    //  |
    //  |
    //  V1(i,j)
    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i-1,j);
    cv::Point2f V3 = m_mesh.getVertex(i-1,j+1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*(i-1) + j;
    int coordv3 = m_meshwidth*(i-1) + j + 1;

    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

void MeshGrid::addCoefficient_4(int i, int j, double weight){
    //          V3 (i-1,j+1)
    //          |
    //          |
    // V1_______|V2

    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i,j+1);
    cv::Point2f V3 = m_mesh.getVertex(i-1,j+1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*i + j + 1;
    int coordv3 = m_meshwidth*(i-1) + j + 1;

    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

void MeshGrid::addCoefficient_5(int i, int j, double weight){
    //  V1     V2
    //  _______
    //          |
    //          |
    //          |
    //          V3
    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i,j+1);
    cv::Point2f V3 = m_mesh.getVertex(i+1,j+1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*i + j + 1;
    int coordv3 = m_meshwidth*(i+1) + j + 1;

    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

void MeshGrid::addCoefficient_6(int i, int j, double weight){
    //  V1
    //  |
    //  |
    //  |_________V3 (i+1,j+1)
    //  V2(i+1,j)
    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i+1,j);
    cv::Point2f V3 = m_mesh.getVertex(i+1,j+1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*(i+1) + j ;
    int coordv3 = m_meshwidth*(i+1) + j + 1;

    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

void MeshGrid::addCoefficient_7(int i, int j, double weight){
    //          V1 (i,j)
    //          |
    //          |
    //  ________|V2
    //  V3(i+1,j-1)
    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i+1,j);
    cv::Point2f V3 = m_mesh.getVertex(i+1,j-1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*(i+1) + j;
    int coordv3 = m_meshwidth*(i+1) + j - 1;

    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

void MeshGrid::addCoefficient_8(int i, int j, double weight){
    //  V2     V1(i,j)
    //  _______
    //  |
    //  |
    //  |
    //  V3(i+1,j-1)
    vector<double> smoothWeight;
    smoothWeight.resize(2);

    cv::Point2f V1 = m_mesh.getVertex(i,j);
    cv::Point2f V2 = m_mesh.getVertex(i,j-1);
    cv::Point2f V3 = m_mesh.getVertex(i+1,j-1);

    getSmoothWeight(smoothWeight,V1,V2,V3);
    double u = smoothWeight[0];
    double v = smoothWeight[1];
    int coordv1 = m_meshwidth*i + j;
    int coordv2 = m_meshwidth*i + j - 1;
    int coordv3 = m_meshwidth*(i+1) + j - 1;

    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
    //////////
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = (1.0-u)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = u*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv2];
    SmoothConstraints.at<double>(Scc,2) = v*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = x_index[coordv3];
    SmoothConstraints.at<double>(Scc,2) = (-1.0*v)*weight;
    Scc++;
    SmoothConstraints.at<double>(Scc,0) = rowCount;
    SmoothConstraints.at<double>(Scc,1) = y_index[coordv1];
    SmoothConstraints.at<double>(Scc,2) = (-1.0)*weight;
    Scc++;
    rowCount++;
}

///////////////////////////////////////////////

void MeshGrid::getGridIndAndWeight(cv::Point2f pt, vector<cv::Point2f> &ind, vector<double> &w){
    int x = (int) floor(pt.x / m_quadWidth);
    int y = (int) floor(pt.y / m_quadHeight);
    ind.resize(4);
    w.resize(4,0.0);

    ind[0].x = x;     ind[0].y = y;
    ind[1].x = x+1;   ind[1].y = y;
    ind[2].x = x+1;   ind[2].y = y+1;
    ind[3].x = x;     ind[3].y = y+1;

    cv::Point2f p0 = m_mesh.getVertex(x, y);
    cv::Point2f p1 = m_mesh.getVertex(x+1, y+1);

    const double xd = pt.x;
    const double yd = pt.y;
    const double xl = p0.x;
    const double xh = p1.x;
    const double yl = p0.y;
    const double yh = p1.y;

    w[0] = (xh - xd) * (yh - yd);
    w[1] = (xd - xl) * (yh - yd);
    w[2] = (xd - xl) * (yd - yl);
    w[3] = (xh - xd) * (yd - yl);
    double s = w[0] + w[1] + w[2] + w[3];
    w[0] = w[0]/s;
    w[1] = w[1]/s;
    w[2] = w[2]/s;
    w[3] = w[3]/s;
}

cv::Mat MeshGrid::WarpingNew(const cv::Mat srcImg){
    cv::Mat warpImg;
    warpImg = cv::Mat::zeros( srcImg.size(), CV_8UC3);

    for(auto y=0; y<srcImg.rows; y++){
        for(auto x=0; x<srcImg.cols; x++){
            vector<cv::Point2f> ind;
            vector<double> w;
            cv::Point2f tmp(x,y);
            getGridIndAndWeight(tmp, ind, w);
            cv::Point2f pt(0,0);
            for(auto i=0; i<4; ++i){
                int tmp_x = ind[i].x;
                int tmp_y = ind[i].y;
                cv::Point2f p0 = m_warpedmesh.getVertex(tmp_x, tmp_y);
                pt.x += p0.x * w[i];
                pt.y += p0.y * w[i];
            }
            if(pt.x < 0 || pt.y < 0 || pt.x > srcImg.cols - 1 || pt.y > srcImg.rows - 1)
                continue;
            pt.x = round(pt.x);
            pt.y = round(pt.y);
            vector<double> pixO = bilinear<uchar,3>(srcImg.data, srcImg.cols, srcImg.rows, pt);
            cv::Vec3b color = srcImg.at<cv::Vec3b>(cv::Point(x,y));
            warpImg.at<cv::Vec3b>(pt.y,pt.x) = color;
//            warpImg.at<cv::Vec3b>(pt.y,pt.x) = cv::Vec3b((uchar) pixO[0], (uchar)pixO[1], (uchar)pixO[2]);
        }
    }

    return warpImg;
}

