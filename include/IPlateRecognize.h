//
// Created by zqp on 18-8-28.
//

#ifndef PROJECT_IPLATERECOGNIZE_H
#define PROJECT_IPLATERECOGNIZE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace Vehicle
{
    using namespace std;

    struct PlateInfo
    {
        cv::Rect zone;
        float score;
        string license;
        string color;
        bool operator ==(PlateInfo const& t) const
        {
            return this->zone==t.zone && this->color==t.color && this->license==t.license;
        }
        bool operator !=(PlateInfo const& t) const
        {
            return this->zone!=t.zone && this->color!=t.color && this->license!=t.license;
        }
    };

    class IPlateRecognize
    {
    public:
        virtual void detect(const cv::Mat &im,vector<PlateInfo>& plateinfos,const float confidence_threshold)=0;
        ~IPlateRecognize(){}
    };

    IPlateRecognize *CreateIPlateRecognize(const string& model_dir,const int gpu_id);

}


#endif //PROJECT_IPLATERECOGNIZE_H
