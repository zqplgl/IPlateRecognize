//
// Created by zqp on 18-8-28.
//

#ifndef PROJECT_PLATERECOGNIZE_H
#define PROJECT_PLATERECOGNIZE_H

#include <IPlateRecognize.h>
#include <IDllPlateDetection2015.h>
namespace Vehicle
{
    class PlateRecognize: public IPlateRecognize
    {
    public:
        PlateRecognize(const string& model_dir,const int gpu_id);
        virtual void detect(const cv::Mat &im,vector<PlateInfo>& plateinfos,const float confidence_threshold);

    private:
        int gpu_id;
        IPlateDetectionInterface *detetor;
    };

}

#endif //PROJECT_PLATERECOGNIZE_H
