struct BA_COST
    {
        
        BA_COST(double real_x,double real_y):_real_x(real_x),_real_y(real_y){};
        template<typename T>
        bool operator()(const T *const point,const T *const rot,const T *const tra,T *residuals)const
        {
            cout<<"point"<<point<<endl;
            cout<<"rot"<<rot<<endl;
            cout<<"tra"<<tra<<endl;
            T theta=sqrt(rot[0]*rot[0]+rot[1]*rot[1]+rot[2]*rot[2]);
            T rx=rot[0]/theta;
            T ry=rot[1]/theta;
            T rz=rot[2]/theta;
            Eigen::Matrix<T,4,4> pose;
            pose(0,0)=T(0.0);
            pose(0,1)=sin(theta)*(-rz);
            pose(0,2)=sin(theta)*(ry);
            pose(0,3)=T(tra[0]);
            pose(1,0)=sin(theta)*(rz);
            pose(1,1)=T(0.0);
            pose(1,2)=sin(theta)*(-rx);
            pose(1,3)=T(tra[1]);
            pose(2,0)=sin(theta)*(-ry);
            pose(2,1)=sin(theta)*(rx);
            pose(2,2)=T(0.0);
            pose(2,3)=T(tra[2]);
            pose(3,0)=T(0.0);
            pose(3,1)=T(0.0);
            pose(3,2)=T(0.0);
            pose(3,3)=T(1.0);
            Eigen::Matrix<T,3,1> r;
            r(0,0)=rx;
            r(1,0)=ry;
            r(2,0)=rz;
            Eigen::Matrix<T,3,3> rr=(T(1.0)-cos(theta))*r*r.transpose();
            for(int i=0;i<3;i++)
                for(int j=0;j<3;j++)
                    pose(i,j)+=rr(i,j);
            for(int i=0;i<3;i++)
                pose(i,i)+=cos(theta);
            pose=pose.inverse().eval();

            Eigen::Matrix<T,3,3> _camMatrix;
            _camMatrix<<T(camMatrix(0,0)),T(camMatrix(0,1)),T(camMatrix(0,2)),
                        T(camMatrix(1,0)),T(camMatrix(1,1)),T(camMatrix(1,2)),
                        T(camMatrix(2,0)),T(camMatrix(2,1)),T(camMatrix(2,2));

            Eigen::Matrix<T,4,1> _point;
            _point<<T(point[0]),T(point[1]),T(point[2]),T(1.0);//传递过来的都是double，记得转换类型
            Eigen::Matrix<T,3,1> point_cam=(pose*_point).block(0,0,3,1);
            Eigen::Matrix<T,3,1> point_cam_norm;
            point_cam_norm<<point_cam(0,0)/point_cam(2,0),point_cam(1,0)/point_cam(2,0),T(1.0);
        
            Eigen::Matrix<T,2,1> point_uv=(_camMatrix*point_cam_norm).block(0,0,2,1);
            residuals[0]=T(point_uv(0,0))-T(_real_x);
            residuals[1]=T(point_uv(1,0))-T(_real_y);
            return 1;
        }
        static ceres::CostFunction* Create(const double _real_x,const double _real_y)
        {
            return (new ceres::AutoDiffCostFunction<BA_COST,2,3,3,3>(
                new BA_COST(_real_x,_real_y)));

        }
        double _real_x,_real_y;
    };
void BA_ceres(std::vector<cv::Point2d> v_Point2D,std::vector<cv::Point3d> v_Point3D,cv::Mat img,Eigen::Vector3d rot,Eigen::Vector3d tra)
{
    
    
    Eigen::MatrixXd x_before;
    int num=v_Point3D.size();
    x_before.resize(num*3,1);
    for(int i=0;i<num;i++)
    {
        x_before(6+3*i,0)=v_Point3D[i].x;
        x_before(6+3*i+1,0)=v_Point3D[i].y;
        x_before(6+3*i+2,0)=v_Point3D[i].z;
    }//状态量初始化完毕
    double* x_ptr = x_before.data();
    auto rvec=rot;
    double* rvec_ptr = rvec.data();
    auto t_copy=tra;
    double* tvec_ptr = t_copy.data();
    
    ceres::Problem problem;
    for (int i = 0; i < num; ++i) {

    ceres::CostFunction* cost_function =
        BA_COST::Create(
            v_Point2D[i].x,
            v_Point2D[i].y);
    problem.AddResidualBlock(cost_function,
                            NULL /* squared loss */,
                            x_ptr+3*i,
                            rvec_ptr,
                            tvec_ptr);
    }
    // 设置优化器参数及优化方法
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 80;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = false;

    // 调用求解器进行优化
    ceres::Solver::Summary summary; // 用于存放日志
    ceres::Solve(options, &problem, &summary); // 构建雅可比矩阵
    std::cout << summary.FullReport() << std::endl;
    Eigen::MatrixXd x;
    x.resize(6+num*3,1);
    x.block(0,0,3,1)=t_copy;
    x.block(3,0,3,1)=rvec;
    x.block(6,0,num*3,1)=x_before;
    

    // Sophus::SE3d se3 = Sophus::SE3d::exp(rvec); // 旋转部分
    // se3.translation() = t_copy; // 设置平移部分
    // Eigen::Matrix<double,4,4> T = se3.matrix();

    // cv::Mat temp_Mat=img.clone();
    //     for(int j=0;j<num;j++)
    //     {
    //         double fx=camMatrix(0,0);
    //         double fy=camMatrix(1,1);
    //         double cx=camMatrix(0,2);
    //         double cy=camMatrix(1,2);
    //         Eigen::Matrix<double,4,1>point;
    //         point(0)=x(6+3*j,0);
    //         point(1)=x(6+3*j+1,0);
    //         point(2)=x(6+3*j+2,0);
    //         point(3)=1;
    //         Eigen::Matrix<double,4,1> cam_Point;
    //         cam_Point=T*point;
    //         cv::Point2d temp_Point2d;
    //         temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
    //         temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
    //         cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
    //         cv::circle(temp_Mat,v_Point2D[j], 2,cv::Scalar(255,0,0),2);
    //     }
    //     imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
    //     cv::waitKey(0);






    

}