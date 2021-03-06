//
// Created by  on 20/09/2017.
//

#ifndef SWIFTPR_PLATEDETECTION_H
#define SWIFTPR_PLATEDETECTION_H

#include <opencv2/opencv.hpp>
#include "PlateInfo.h"
#include <vector>
namespace pr{
    class PlateDetection{
    public:
        PlateDetection(std::string filename_cascade);
        void plateDetectionRough(cv::Mat InputImage,std::vector<pr::PlateInfo>  &plateInfos,int min_w=36,int max_w=800);
       

    private:
        cv::CascadeClassifier cascade;

    };

}// namespace pr

#endif //SWIFTPR_PLATEDETECTION_H
