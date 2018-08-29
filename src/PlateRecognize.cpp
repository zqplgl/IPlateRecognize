//
// Created by zqp on 18-8-28.
//
#include <IDllPlateDetection2015.h>
#include <PlateRecognize.h>

namespace Vehicle
{
    PlateRecognize::PlateRecognize(const std::string &model_dir, const int gpu_id):gpu_id(gpu_id)
    {
        detetor = CreateDetector(model_dir);
    }

    void PlateRecognize::detect(const cv::Mat &im, std::vector<Vehicle::PlateInfo> &plateinfos,const float confidence_threshold)
    {
        plateinfos.clear();
        pr::PlateInfo plate = detetor->PlateRecogPipeline(im.data,im.cols,im.rows,confidence_threshold);

        if(plate.confidence<confidence_threshold || plate.name.size()<4)
            return;
        Vehicle::PlateInfo plateinfo;
        plateinfo.zone = plate.ROI;
        plateinfo.license = plate.name;
        plateinfo.score = plate.confidence;

        switch (plate.ColorType)
        {
            case BLUE:
                plateinfo.color = "blue";
                break;
            case YELLOW:
                plateinfo.color = "yellow";
                break;
            case GREEN:
                plateinfo.color = "green";
                break;
            case BLACK:
                plateinfo.color = "black";
                break;
            case WHITE:
                plateinfo.color = "white";
                break;
            default:
                plateinfo.color = "unknown";
        }

        plateinfos.push_back(plateinfo);
    }

    IPlateRecognize *CreateIPlateRecognize(const string& model_dir,const int gpu_id)
    {
        return new PlateRecognize(model_dir,gpu_id);
    }
}
