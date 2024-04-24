#include <vector>
#include <iostream>
#include <stdio.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "sophus/se3.hpp"

#include <ceres/ceres.h>
#include "ceres/rotation.h"
using namespace cv;
using namespace std;
//定义相机内参矩阵
Eigen::Matrix<double,3,3> camMatrix;
//定义最大迭代次数
const int MAX_LOOP=5;


//构造相机外姿雅可比
Eigen::Matrix<double,2,6> find_jacobian_pose(Eigen::Vector3d point,Eigen::Matrix<double,4,4> pose)
{
    double fx=camMatrix(0,0);
    double fy=camMatrix(1,1);
    Eigen::Matrix<double,4,1> temp;
    temp<<point(0),point(1),point(2),1;
    Eigen::Matrix<double,4,1> point_cam;
    point_cam=pose*temp;
    double X=point_cam(0);
    double Y=point_cam(1);
    double Z=point_cam(2);
    double Z2=Z*Z;
    Eigen::Matrix<double,2,6> jacobian;
    jacobian<<fx/Z,0,-fx*X/Z2,-fx*X*Y/Z2,fx+fx*X*X/Z2,-fx*Y/Z,
              0,fy/Z,-fy*Y/Z2,-fy-fy*Y*Y/Z2,fy*X*Y/Z2,fy*X/Z;
    return -1.0*jacobian;
}
Eigen::MatrixXd find_jacobian_point(Eigen::Vector3d point,Eigen::Matrix<double,4,4> pose)
{
    double fx=camMatrix(0,0);
    double fy=camMatrix(1,1);
    Eigen::Matrix<double,4,1> temp;
    temp<<point(0),point(1),point(2),1;
    Eigen::Matrix<double,4,1> point_cam;
    point_cam=pose*temp;
    double X=point_cam(0);
    double Y=point_cam(1);
    double Z=point_cam(2);
    double Z2=Z*Z;
    Eigen::Matrix<double,2,3> jacobian;
    jacobian<<fx/Z,0,-fx*X/Z2,
              0,fy/Z,-fy*Y/Z2;
    return -1.0*jacobian*pose.block<3,3>(0,0);

}
Eigen::MatrixXd find_jacobian_all(Eigen::MatrixXd x)
{
    Eigen::MatrixXd jacobian;
    int num=(x.size()-6)/3;//求出待优化特征点的数量
    jacobian.resize(2*num,6+3*num);//最终的雅可比大小
    jacobian.setZero();
    auto pose_temp=Sophus::SE3d::exp(x.block(0,0,6,1));//先把相机位姿转换过来
    Eigen::Matrix<double,4,4> pose=pose_temp.matrix();//记得转换格式
    for(int i=0;i<num;i++)
    {

        jacobian.block(2*i,0,2,6)=find_jacobian_pose(x.block(6+i*3,0,3,1),pose);
        jacobian.block(2*i,6+3*i,2,3)=find_jacobian_point(x.block(6+i*3,0,3,1),pose);//每次是要按顺序错开一定位置的

    }
    return jacobian;
    
}
Eigen::MatrixXd find_costfunction(Eigen::MatrixXd x,std::vector<cv::Point2d> v_Point2D)
{
    auto pose_temp=Sophus::SE3d::exp(x.block(0,0,6,1));//先把相机位姿转换过来
    Eigen::Matrix<double,4,4> pose=pose_temp.matrix();//记得转换格式
    int num=(x.size()-6)/3;//求出待优化特征点的数量
    Eigen::MatrixXd error;
    error.resize(2*num,1);
    for(int i=0;i<num;i++)
    {
        Eigen::Vector3d point=x.block(6+i*3,0,3,1);
        Eigen::Matrix<double,4,1> temp;
        temp<<point(0),point(1),point(2),1;
        Eigen::Matrix<double,4,1> point_cam;
        point_cam=pose*temp;
        Eigen::Vector3d point_norm;
        point_norm<<point_cam(0)/point_cam(2),point_cam(1)/point_cam(2),1;
        auto temp_uv=camMatrix*point_norm;
        Eigen::Vector2d point_uv;
        point_uv<<temp_uv(0),temp_uv(1);
        Eigen::Vector2d error_temp;
        error_temp(0)=v_Point2D[i].x-point_uv(0);
        error_temp(1)=v_Point2D[i].y-point_uv(1);
        error.block(2*i,0,2,1)=error_temp;
    }
    return error;
}
void has_NaN(Eigen::MatrixXd matrix)
{
    bool hasNaN = matrix.array().isNaN().any();
    if (hasNaN) {
        std::cout << "Matrix contains NaN values." << std::endl;
    };

}
void has_INF(Eigen::MatrixXd mat)
{
     bool hasInf = false;

    // 遍历矩阵并检查是否有 INF 值
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            if (std::isinf(mat(i, j))) {
                hasInf = true;
                break;
            }
        }
        if (hasInf) {
            break;
        }
    }

    if (hasInf) {
        std::cout << "矩阵中存在 INF 值！" << std::endl;
    }

}
Eigen::MatrixXd INF_trans(Eigen::MatrixXd mat)
{
     bool hasInf = false;

    // 遍历矩阵并检查是否有 INF 值
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            if (std::isinf(mat(i, j))) {
                hasInf = true;
                mat(i, j)=1e300;
            }
        }
    }

    if (hasInf) {
        std::cout << "矩阵中存在 INF 值！" << std::endl;
    }
    return mat;

}
double find_beta(Eigen::MatrixXd param_a,Eigen::MatrixXd param_b,double u)
{
    auto diff=param_b-param_a;
    double lower=-1,upper=1,epsilon=1e-4,beta=0;
    while(upper-lower>=epsilon)
    {
        beta=(lower+upper)/2.0;//取中间值
        auto temp=param_a+beta*diff;
        double L1_norm=temp.lpNorm<1>();

        if(L1_norm<u)
        {
            lower=beta;
        }
        else
        {
            upper=beta;
        }

    }
    return beta;
}

void my_BA_DogLeg(std::vector<cv::Point2d> v_Point2D,std::vector<cv::Point3d> v_Point3D,cv::Mat img,Eigen::Matrix<double,4,4> T)
{
    Eigen::MatrixXd h_sd,h_gn,h_dl;//决定最终的下降方向
    double alpha,u=0.5,beta;
    Eigen::MatrixXd update;
    Eigen::Matrix3d R=T.block(0,0,3,3);
    Eigen::Vector3d t=T.block(0,3,3,1);
    Sophus::SE3d SE3_Rt(R,t);
    Eigen::Matrix<double,6,1> se3=SE3_Rt.log();
    Eigen::MatrixXd x;
    int num=v_Point3D.size();
    x.resize(6+num*3,1);
    x.block(0,0,6,1)=se3;
    for(int i=0;i<num;i++)
    {
        x(6+3*i,0)=v_Point3D[i].x;
        x(6+3*i+1,0)=v_Point3D[i].y;
        x(6+3*i+2,0)=v_Point3D[i].z;
    }//状态量初始化完毕
    for(int iter=0;iter<MAX_LOOP;iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now(); // 记录程序开始时间
        double beta,curSquareCost,curSquareCost_aft,taylar_similary;//u为初始半径其实也就是三角形的delta
        Eigen::MatrixXd H,b;
        
        
        
        Eigen::MatrixXd J=find_jacobian_all(x);
        H=J.transpose()*J;
        Eigen::MatrixXd error=find_costfunction(x,v_Point2D);
        b=-J.transpose()*error;
        alpha=pow(b.lpNorm<1>(),2)/pow((J*b).lpNorm<1>(),2);
        h_sd=-alpha*b;
        h_gn=H.colPivHouseholderQr().solve(b); 

        //计算公式中的beta
        Eigen::MatrixXd param_a=alpha*h_sd;
        Eigen::MatrixXd param_b=h_gn;
        beta=find_beta(param_a,param_b,u);
        // auto param_c = param_a.transpose() * (param_b - param_a);
		// if (param_c<=0) {
 
		// 	beta = (sqrt( pow(param_c,2) + pow((param_b-param_a).lpNorm<1>(), 2)  *
		// 							( pow(u,2)- pow(param_a.lpNorm<1>(),2) ) ) - param_c) / 
		// 											pow((param_b-param_a).lpNorm<1>(), 2);
		// }else {
 
		// 	beta = ( pow(u,2)- pow(param_a.lpNorm<1>(),2) ) / 
		// 					(sqrt( pow(param_c,2) + pow((param_b-param_a).lpNorm<1>(), 2) *
		// 							( pow(u,2)- pow(param_a.lpNorm<1>(),2) ) ) + param_c);
		// }

        // 计算h_dl
        if(param_b.lpNorm<1>()<=u)
        {
            h_dl=h_gn;//高斯解如果满足约束则用高斯牛顿
        }
        else if(param_a.lpNorm<1>()>=u)
        {
            h_dl=(u/(h_sd.lpNorm<1>()))*h_sd;// 高斯解与最速下降的解都在约束外，用最速（更改步进）
        }
        else
        {
            h_dl=param_a+beta*(param_b-param_a);//高斯解在约束外，最速解在约束内，两者融合
        }
        //提前终止迭代的条件省略了
        curSquareCost=1.0f/2*error.squaredNorm();

        auto temp_x=x;//LM最关键的回滚
        auto update=h_dl;
        Eigen::Matrix<double,6,1> pose_update=update.block(0,0,6,1);
        Eigen::Matrix<double,6,1> pose_last=temp_x.block(0,0,6,1);
        auto temp_update=Sophus::SE3d::exp(pose_update);
        auto temp_last=Sophus::SE3d::exp(pose_last);
        auto temp_after_update=temp_update*temp_last;
        Eigen::Matrix<double,6,1> pose_after_update=temp_after_update.log();
        temp_x+=update;
        temp_x.block(0,0,6,1)=pose_after_update;
        Eigen::Matrix<double,4,4> pose= (Sophus::SE3d::exp(pose_after_update)).matrix();

        
        
        Eigen::MatrixXd error_new=find_costfunction(temp_x,v_Point2D);
        curSquareCost_aft=1.0f/2*error_new.squaredNorm();

        Eigen::MatrixXd L0_Ldelta = -h_dl.transpose() * J.transpose() * error - 1.0f/2 * h_dl.transpose() * J.transpose() * J * h_dl;
		Eigen::MatrixXd L0_Ldelta2 = 1.0f/2 * h_dl.transpose() * ( u*h_dl + b);  
		Eigen::MatrixXd L0_Ldelta3 = 1.0/2*((J*h_dl).transpose() * (J*h_dl));//这三种分别代表不同的意义
		
		taylar_similary = (curSquareCost-curSquareCost_aft) / L0_Ldelta(0,0); 
		// 信赖域更新法则
		if (taylar_similary>0) { 	
			// 更新增量
			x=temp_x;		
		}
 
		if (taylar_similary>0.75) {
 
			u = std::max<double>(u, 3.0*h_dl.lpNorm<1>() );	
		} else if (taylar_similary<0.25) { 
 
			u = u/2.0;	
		} else {
 
			u = u;
		}
        auto end_time = std::chrono::high_resolution_clock::now(); // 记录程序结束时间


        // 计算程序运行时间
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "BAbyDogLeg optimization completed in " << elapsed.count() << " seconds." << std::endl;


        cv::Mat temp_Mat=img.clone();
        for(int j=0;j<num;j++)
        {
            double fx=camMatrix(0,0);
            double fy=camMatrix(1,1);
            double cx=camMatrix(0,2);
            double cy=camMatrix(1,2);
            Eigen::Matrix<double,4,1>point;
            point(0)=x(6+3*j,0);
            point(1)=x(6+3*j+1,0);
            point(2)=x(6+3*j+2,0);
            point(3)=1;
            Eigen::Matrix<double,4,1> cam_Point;
            cam_Point=pose*point;
            cv::Point2d temp_Point2d;
            temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
            temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
            cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
            cv::circle(temp_Mat,v_Point2D[j],    2,cv::Scalar(255,0,0),2);
        }
        imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
        cout<<"\033[32m"<<"Iteration： "<<iter<<" Finish......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Blue is observation......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Red is reprojection......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Press Any Key to continue......"<<"\033[37m"<<endl;
        cv::waitKey(0);
        
        
    }

}
void my_BA_GN(std::vector<cv::Point2d> v_Point2D,std::vector<cv::Point3d> v_Point3D,cv::Mat img,Eigen::Matrix<double,4,4> T)//T的初始值随便搞
{
    // Eigen::MatrixXd B,E,v,w,C,C_inv,ET,left,right,delta_xc,delta_xp;
    
    Eigen::MatrixXd update;
    Eigen::Matrix3d R=T.block(0,0,3,3);
    Eigen::Vector3d t=T.block(0,3,3,1);
    Sophus::SE3d SE3_Rt(R,t);
    Eigen::Matrix<double,6,1> se3=SE3_Rt.log();
    Eigen::MatrixXd x;
    int num=v_Point3D.size();
    x.resize(6+num*3,1);
    x.block(0,0,6,1)=se3;
    for(int i=0;i<num;i++)
    {
        x(6+3*i,0)=v_Point3D[i].x;
        x(6+3*i+1,0)=v_Point3D[i].y;
        x(6+3*i+2,0)=v_Point3D[i].z;
    }//状态量初始化完毕
    double cost=0,lastcost=0;
    for(int iter=0;iter<MAX_LOOP;iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now(); // 记录程序开始时间
        
        Eigen::MatrixXd H;
        Eigen::MatrixXd b;
        
        cost=0;
        Eigen::MatrixXd J=find_jacobian_all(x);
        H=J.transpose()*J;
        Eigen::MatrixXd error=find_costfunction(x,v_Point2D);
        b=-J.transpose()*error;



        // B=H.block(0,0,6,6);
        // C=H.block(6,6,192,192);
        // E=H.block(6,0,6,192);
        // ET=E.transpose();
        // v=b.block(0,0,6,1);
        // w=b.block(6,0,192,1);
        // C_inv.resize(192,192);
        // update.resize(198,1);



        // for (int i = 0; i < 192; i += 3) 
        // {
            
        //         Eigen::MatrixXd block = H.block(i, i, 3, 3);
        //         if(block.determinant()!=0)
        //         {
        //             Eigen::MatrixXd block_inv = block.inverse();
                    
        //             C_inv.block(i, i, 3, 3) = block_inv;

        //         }
        //         else
        //         {
        //             // cout<<"C"<<block<<endl;
        //             Eigen::MatrixXd C_pseudo_inv = block.completeOrthogonalDecomposition().pseudoInverse();
                    
        //             C_inv.block(i, i, 3, 3)=C_pseudo_inv;
        //         }
                
            
        // }
        
        
        // left=B-(E*C_inv*ET);
        // cout<<"C_inv*ET is"<<C_inv*ET<<endl;
        // right=v-E*C_inv*w;
        // cout<<"left.inverse() is"<<left.inverse()<<endl;
        // cout<<"right is"<<right<<endl;
        // delta_xc=left.inverse()*right;
        // delta_xp=C_inv*(w-ET*delta_xc);
        // cout<<"delta_xc.size"<<delta_xc.rows()<<endl;
        // cout<<"delta_xc is"<<delta_xc<<endl;
        // cout<<"delta_xp.size"<<delta_xp.rows()<<endl;
        // cout<<"delta_xp is"<<delta_xp<<endl;
        // update.block(0,0,6,1)=delta_xc;
        // update.block(6,1,192,1)=delta_xp;
        // cout<<"C矩阵的左上角"<<C.block(0,0,15,15);

        // cout<<"update="<<update.rows()<<endl;


        
        
        
        update=H.colPivHouseholderQr().solve(b);
        Eigen::Matrix<double,6,1> pose_update=update.block(0,0,6,1);
        Eigen::Matrix<double,6,1> pose_last=x.block(0,0,6,1);
        auto temp_update=Sophus::SE3d::exp(pose_update);
        auto temp_last=Sophus::SE3d::exp(pose_last);
        auto temp_after_update=temp_update*temp_last;
        Eigen::Matrix<double,6,1> pose_after_update=temp_after_update.log();
        x+=update;
        x.block(0,0,6,1)=pose_after_update;
        Eigen::Matrix<double,4,4> pose= (Sophus::SE3d::exp(pose_after_update)).matrix();
        auto end_time = std::chrono::high_resolution_clock::now(); // 记录程序结束时间


        // 计算程序运行时间
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "BAbyGN optimization completed in " << elapsed.count() << " seconds." << std::endl;


        cv::Mat temp_Mat=img.clone();
        for(int j=0;j<num;j++)
        {
            double fx=camMatrix(0,0);
            double fy=camMatrix(1,1);
            double cx=camMatrix(0,2);
            double cy=camMatrix(1,2);
            Eigen::Matrix<double,4,1>point;
            point(0)=x(6+3*j,0);
            point(1)=x(6+3*j+1,0);
            point(2)=x(6+3*j+2,0);
            point(3)=1;
            Eigen::Matrix<double,4,1> cam_Point;
            cam_Point=pose*point;
            cv::Point2d temp_Point2d;
            temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
            temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
            cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
            cv::circle(temp_Mat,v_Point2D[j],    2,cv::Scalar(255,0,0),2);
        }
        imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
        cout<<"\033[32m"<<"Iteration： "<<iter<<" Finish......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Blue is observation......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Red is reprojection......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Press Any Key to continue......"<<"\033[37m"<<endl;
        cv::waitKey(0);
    }
}
// Eigen::MatrixXd roll_back(Eigen::Matrix<double,6,1> update,Eigen::MatrixXd x)
// {
//     Eigen::Matrix<double,6,1> pose_update=update.block(0,0,6,1);
//     Eigen::Matrix<double,6,1> pose_after=x.block(0,0,6,1);
//     auto temp_after=Sophus::SE3d::exp(pose_after);
//     auto temp_update=Sophus::SE3d::exp(pose_update);
//     auto temp_before_update=temp_after*temp_update;
//     Eigen::Matrix<double,6,1> pose_after_update=temp_before_update.log();
//     x-=update;
//     x.block(0,0,6,1)=pose_after_update;
//     return x;
// }
void my_BA_LM(std::vector<cv::Point2d> v_Point2D,std::vector<cv::Point3d> v_Point3D,cv::Mat img,Eigen::Matrix<double,4,4> T)//T的初始值随便搞
{
    double tao=1e-6,p=0,v=2,cost=0,cost_new=0,diag_max=0,lamda=0;
    int iter=0,final_iter=0;
    bool flag=0;
    Eigen::Matrix3d R=T.block(0,0,3,3);
    Eigen::Vector3d t=T.block(0,3,3,1);
    Sophus::SE3d SE3_Rt(R,t);
    Eigen::Matrix<double,6,1> se3=SE3_Rt.log();
    Eigen::MatrixXd x;
    int num=v_Point3D.size();
    x.resize(6+num*3,1);
    x.block(0,0,6,1)=se3;
    for(int i=0;i<num;i++)
    {
        x(6+3*i,0)=v_Point3D[i].x;
        x(6+3*i+1,0)=v_Point3D[i].y;
        x(6+3*i+2,0)=v_Point3D[i].z;
    }//状态量初始化完毕
    
    
    for( iter=0;iter<MAX_LOOP;iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now(); // 记录程序开始时间
        Eigen::MatrixXd H;
        Eigen::MatrixXd b;
        Eigen::MatrixXd J=find_jacobian_all(x);
        H=J.transpose()*J;
        Eigen::MatrixXd error=find_costfunction(x,v_Point2D);
        cost=error.norm();
        b=-J.transpose()*error;
        static bool isFirstStart = true;

        if (isFirstStart) {
        diag_max=0;
        for(int i=0;i<H.rows();i++)
        {
        if(diag_max<H(i,i))
        {
            diag_max=H(i,i);
        }
        isFirstStart = false;
        lamda=diag_max*tao;
        }}//一般选取对角线上最大元素设定初始值
        
        int H_rows=H.rows();
        int H_cols=H.cols();
        Eigen::MatrixXd I;
        I.resize(H_rows,H_cols);
        I.setIdentity();
        
        Eigen::MatrixXd A=H+lamda*I;
        Eigen::MatrixXd update=H.colPivHouseholderQr().solve(b);

        auto temp_x=x;//LM最关键的回滚
        Eigen::Matrix<double,6,1> pose_update=update.block(0,0,6,1);
        Eigen::Matrix<double,6,1> pose_last=temp_x.block(0,0,6,1);
        auto temp_update=Sophus::SE3d::exp(pose_update);
        auto temp_last=Sophus::SE3d::exp(pose_last);
        auto temp_after_update=temp_update*temp_last;
        Eigen::Matrix<double,6,1> pose_after_update=temp_after_update.log();
        temp_x+=update;
        temp_x.block(0,0,6,1)=pose_after_update;

        
        
        error=find_costfunction(temp_x,v_Point2D);
        cost_new=error.norm();
        
        p=(cost-cost_new)/(update.transpose()*(lamda*update+b))(0,0);//参考其他公式，记得加（0，0）
        
        
        

        if(p>0)//说明误差确实在下降
            {
                x=temp_x;//如果满足条件则更新
                lamda=lamda*max(0.3333333,(1-(2*p-1))*(2*p-1)*(2*p-1));
                v=2;
            }
            else
            {
                lamda=lamda*v;
                v=2*v;
            }
            //lastcost=cost;

        Eigen::Matrix<double,4,4> pose= (Sophus::SE3d::exp(pose_after_update)).matrix();
        auto end_time = std::chrono::high_resolution_clock::now(); // 记录程序结束时间

        // 计算程序运行时间
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "BAbyLM optimization completed in " << elapsed.count() << " seconds." << std::endl;


        cv::Mat temp_Mat=img.clone();
        for(int j=0;j<num;j++)
        {
            double fx=camMatrix(0,0);
            double fy=camMatrix(1,1);
            double cx=camMatrix(0,2);
            double cy=camMatrix(1,2);
            Eigen::Matrix<double,4,1>point;
            point(0)=x(6+3*j,0);
            point(1)=x(6+3*j+1,0);
            point(2)=x(6+3*j+2,0);
            point(3)=1;
            Eigen::Matrix<double,4,1> cam_Point;
            cam_Point=pose*point;
            cv::Point2d temp_Point2d;
            temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
            temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
            cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
            cv::circle(temp_Mat,v_Point2D[j],    2,cv::Scalar(255,0,0),2);
        }
        imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
        cout<<"\033[32m"<<"Iteration： "<<iter<<" Finish......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Blue is observation......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Red is reprojection......"<<"\033[37m"<<endl;
        cout<<"\033[32m"<<"Press Any Key to continue......"<<"\033[37m"<<endl;
        cv::waitKey(0);
    }
    
}
//图优化解BA
void BA_G2o(std::vector<cv::Point2d> v_Point2D,std::vector<cv::Point3d> v_Point3D,cv::Mat img,Eigen::Matrix<double,3,3> R,Eigen::Vector3d t)
{
    
  
    Eigen::MatrixXd x;
    int num=v_Point3D.size();
    x.resize(6+num*3,1);
    for(int i=0;i<num;i++)
    {
        x(6+3*i,0)=v_Point3D[i].x;
        x(6+3*i+1,0)=v_Point3D[i].y;
        x(6+3*i+2,0)=v_Point3D[i].z;
    }//状态量初始化完毕
    
    
    //step1 初始化
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
    Block::LinearSolverType* linearSolver =new g2o::LinearSolverCSparse<Block::PoseMatrixType>();//初始化线性求解器PoseMatrixType，Block::PoseMatrixType 表示优化问题中位姿的矩阵类型。
    Block* solver_ptr =new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));//注意增加智能指针！！！
    // g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);//选择优化算法
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));//注意增加智能指针！！！
    g2o::SparseOptimizer optimizer;//创建一个稀疏优化器
    optimizer.setAlgorithm(solver);//设置了优化器的算法，将前面创建的优化算法传递给优化器。
    //step2 相机姿态
    g2o::VertexSE3Expmap *pose=new g2o::VertexSE3Expmap();//相机姿态初始化
    cout<<"init_R is "<<R<<endl;
    cout<<"init_t is "<<t<<endl;
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R,t));//必须使用g2o的SE3Quat，其中必须旋转在前，平移在后
    optimizer.addVertex(pose);//设置优化器顶点0
    //step3设置优化器顶点1与之后的（特征点landmark）
    int index=1;
    for(const Point3f p:v_Point3D)//遍历特征点
    {
        g2o::VertexSBAPointXYZ *point=new g2o::VertexSBAPointXYZ();
        point->setId(index++);//索引对应增加
        point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));//设置由特征点1在图一像素坐标由深度图1反推出的3D坐标,然后把特征点2作为mearsure,设定一个单位矩阵外参T，不断优化T使投影到图二上符合measure
        point->setMarginalized(true);//一般边缘化特征点
        optimizer.addVertex(point);
    }

    //step4设定相机内参
    g2o::CameraParameters* camera=new g2o::CameraParameters(camMatrix(0,0),Eigen::Vector2d(camMatrix(0,2),camMatrix(1,2)),0);//设置方式是先焦距，默认fx=fy?,然后是相机中心点的偏移(内参矩阵中的偏移量)，通常与相机的平移向量有关。最后是畸变参数的数量与...
    camera->setId(0);
    optimizer.addParameter(camera);//设置优化器参数
    //step5加边
    index=1;//一定要记得把index重置，否则边就加空了
    for(const Point2f p:v_Point2D)
    {
        g2o::EdgeProjectXYZ2UV*edge=new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));//将特征点类型转换为顶点类型
        edge->setVertex(1,pose);
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 5 );//迭代100次
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
    Eigen::Matrix<double,4,4> T=Eigen::Isometry3d ( pose->estimate() ).matrix();
    for(int i=1;i<=num;i++)
    {g2o::VertexSBAPointXYZ* optimized_point = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i));

    if (optimized_point != nullptr) {
        Eigen::Vector3d updated_point = optimized_point->estimate();
        x.block(6+3*(i-1),0,3,1)=updated_point;
        // 使用更新后的特征点位置信息 updated_point
    }}

    
    Eigen::Matrix3d R_after=T.block(0,0,3,3);
    Eigen::Vector3d t_after=T.block(0,3,3,1);
    Sophus::SE3d SE3_Rt(R_after,t_after);
    Eigen::Matrix<double,6,1> se3=SE3_Rt.log();
    x.block(0,0,6,1)=se3;
    cout<<"T_G2o"<<endl<<T;

    cv::Mat temp_Mat=img.clone();
    num=(x.size()-6)/3;
        for(int j=0;j<num;j++)
        {
            double fx=camMatrix(0,0);
            double fy=camMatrix(1,1);
            double cx=camMatrix(0,2);
            double cy=camMatrix(1,2);
            Eigen::Matrix<double,4,1>point;
            point(0)=x(6+3*j,0);
            point(1)=x(6+3*j+1,0);
            point(2)=x(6+3*j+2,0);
            point(3)=1;
            Eigen::Matrix<double,4,1> cam_Point;
            cam_Point=T*point;
            cv::Point2d temp_Point2d;
            temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
            temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
            cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
            cv::circle(temp_Mat,v_Point2D[j], 2,cv::Scalar(255,0,0),2);
        }
        imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
        cv::waitKey(0);
    
}
struct BA_COST
    {
        
        BA_COST(double real_x,double real_y):_real_x(real_x),_real_y(real_y){};
        template<typename T>
        bool operator()(const T *const point,const T *const rot,const T *const tra,T *residuals)const
        {
            T point_cam[3];
            T point_[3];
            T rot_[3];
            for(int i=0;i<3;i++)
            {
                point_[i]=T(point[i]);
                rot_[i]=T(rot[i]);//AngleAxisRotatePoint函数经测试之后接受数组类型的数据
            }
            ceres::AngleAxisRotatePoint(rot, point, point_cam);//只考虑旋转（向量）的投影
            point_cam[0]+=T(tra[0]);
            point_cam[1]+=T(tra[1]);
            point_cam[2]+=T(tra[2]);//加上平移向量



            Eigen::Matrix<T,3,3> _camMatrix;
            _camMatrix<<T(camMatrix(0,0)),T(camMatrix(0,1)),T(camMatrix(0,2)),
                        T(camMatrix(1,0)),T(camMatrix(1,1)),T(camMatrix(1,2)),
                        T(camMatrix(2,0)),T(camMatrix(2,1)),T(camMatrix(2,2));

           
            Eigen::Matrix<T,3,1> point_cam_norm;
            point_cam_norm<<point_cam[0]/point_cam[2],point_cam[1]/point_cam[2],T(1.0);
        
            Eigen::Matrix<T,2,1> point_uv=(_camMatrix*point_cam_norm).block(0,0,2,1);
            residuals[0]=T(point_uv(0,0))-T(_real_x);
            residuals[1]=T(point_uv(1,0))-T(_real_y);
            return true;
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
        x_before(3*i,0)=v_Point3D[i].x;
        x_before(3*i+1,0)=v_Point3D[i].y;
        x_before(3*i+2,0)=v_Point3D[i].z;
    }//状态量初始化完毕
    
    double* x_ptr = x_before.data();
    auto rvec =rot;
    double* rvec_ptr = rvec.data();
    auto t_copy=tra;
    double* tvec_ptr = t_copy.data();//记得重载运算符时传入的参数必须是指针
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
     //设置优化器参数及优化方法
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
 
 
    //调用求解器进行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    Eigen::MatrixXd x;
    x.resize(6+num*3,1);
    x.block(0,0,3,1)=t_copy;
    x.block(3,0,3,1)=rvec;
    x.block(6,0,num*3,1)=x_before;

    // cout<<"rvec is"<<rvec<<endl;
    // cout<<"t_copy is"<<t_copy<<endl;

    // Eigen::AngleAxisd rotation_matrix(rvec.norm(), rvec.normalized()); // AngleAxis表示旋转
    // Eigen::Matrix3d rotation_matrix_final = rvec.matrix(); // 转换为旋转矩阵
    // Eigen::Matrix<double,4,4> T;
    // T.setIdentity();
    // T.block(0,0,3,3)=rotation_matrix_final;
    // T.block(3,0,3,1)=t_copy;



    Eigen::Matrix<double,6,1> temp=x.block(0,0,6,1);
    Eigen::Matrix<double,4,4> T=Sophus::SE3d::exp(temp).matrix();
    cout<<"T_ceres"<<endl<<T;

    
    cv::Mat temp_Mat=img.clone();
        for(int j=0;j<num;j++)
        {
            double fx=camMatrix(0,0);
            double fy=camMatrix(1,1);
            double cx=camMatrix(0,2);
            double cy=camMatrix(1,2);
            Eigen::Matrix<double,4,1>point;
            point(0)=x(6+3*j,0);
            point(1)=x(6+3*j+1,0);
            point(2)=x(6+3*j+2,0);
            point(3)=1.0;
            Eigen::Matrix<double,4,1> cam_Point;
            cam_Point=T*point;
            cv::Point2d temp_Point2d;
            temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
            temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
            cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
            cv::circle(temp_Mat,v_Point2D[j], 2,cv::Scalar(255,0,0),2);
        }
        imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
        cv::waitKey(0);






    

}
int main(int argc, char  *argv[])
{
    if(argc!=4) {
        cout<<"\033[31m"<<"INPUT ERROR !"<<"\033[32m"<<"Please Input Like: 1.png 2.png 3.png  --(1 is the first image,2 is the second ,3 is depth image)"<<"\033[37m"<<endl;
        return -1;
    }
    string imag1 = argv[1];
    string imag2 = argv[2];
    string imag3 = argv[3];//接收三个图片的路径
    //-- 设置相机内参
    camMatrix(0,0)=525.0;
    camMatrix(1,1)=525.0;
    camMatrix(0,2)=319.5;
    camMatrix(1,2)=239.5;
    camMatrix(2,2)=1.0;
    //-- 读取图像
    Mat img_1 = imread (imag1);
    Mat img_2 = imread (imag2);
    Mat img_depth = imread(imag3);
    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create(500);
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第一步:检测 Oriented FAST 角点位置
    double t = (double)cv::getTickCount();
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );
    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    matcher->match ( descriptors_1, descriptors_2, matches );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
        
    }
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }
    

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    printf("ORB detect cost %f ms \n", (1000*(cv::getTickCount() - t) / cv::getTickFrequency()));
    cout<<"good_match = "<<good_matches.size()<<endl;

    std::vector<cv::Point2d> points1,points2;
    std::vector< DMatch > good_matches2;
    std::vector<cv::Point3d> v_3DPoints;
    for ( int i = 0; i < ( int ) good_matches.size(); i++ )
    {
        ///
        if(img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)/5000!=0 &&
         !isnan( img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)) &&
         !isinf( img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)))
        {
            points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
            good_matches2.emplace_back(good_matches[i]);

            //求解3D点坐标，参见TUM数据集
            cv::Point3d temp;
            double  u=keypoints_1[good_matches[i].queryIdx].pt.x;
            double v=keypoints_1[good_matches[i].queryIdx].pt.y;
            temp.z=img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)/5000;
            //if(temp.z==0) temp.z = 1;
            temp.x=(u-camMatrix(0,2))*temp.z/camMatrix(0,0);
            temp.y=(v-camMatrix(1,2))*temp.z/camMatrix(1,1);
            v_3DPoints.emplace_back(temp);//得到相机坐标系下的坐标
        }
    }
    //cout<<"points1.size:"<<points1.size()<<endl;
    //cout<<"points2.size:"<<points2.size()<<endl;
    cout<<"v_3DPoints.size:"<<v_3DPoints.size()<<endl;
    //使用OpenCV提供的代数方法求解：2D-2D
    Point2d principal_point ( 319.5, 239.5);	//相机光心, TUM dataset标定值
    double focal_length = 525;			        //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cv::Mat R,tt;
    recoverPose ( essential_matrix, points1, points2, R, tt, focal_length, principal_point );
    Eigen::Matrix<double,3,3> R_init;
    Eigen::Matrix<double,3,1> tt_init;
    // R_init <<
    //            R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
    //            R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
    //            R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    // tt_init<<tt.at<double>(0,0),tt.at<double>(1,0),tt.at<double>(2,0);
    R_init.setIdentity();
    tt_init<<0,0,0;
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<tt<<endl;
    // -- 开始BA优化求解：3D-2D
    Eigen::Matrix<double,4,4> init;
    init.setIdentity();
    Eigen::Vector3d rot,tra;
    my_BA_GN(points2,v_3DPoints,img_2,init);
    my_BA_LM(points2,v_3DPoints,img_2,init);
    my_BA_DogLeg(points2,v_3DPoints,img_2,init);
    BA_G2o(points2,v_3DPoints,img_2,R_init,tt_init);
    rot<<0.01,0.01,0.01;
    tra<<0.01,0.01,0.01;
    BA_ceres(points2,v_3DPoints,img_2,rot,tra);

    return 0;
}

//命令 ./BA 11.png 33.png 111.png