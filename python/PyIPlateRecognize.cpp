//
// Created by zqp on 18-8-1.
//
#include<boost/python.hpp>
#include <numpy/arrayobject.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <PlateRecognize.h>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace boost::python;
using namespace Vehicle;
using namespace cv;

class PlateDetector
{
public:
    PlateDetector(const string &model_dir,const int gpu_id)
    {
         detector = CreateIPlateRecognize(model_dir,gpu_id);
    }

    vector<Vehicle::PlateInfo> detect(boost::python::object &data_obj, int w, int h,const float confidence_threshold)
    {
        PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
        unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(data_arr));

        Mat im(h,w,CV_8UC3,data);
        detector->detect(im,r,confidence_threshold);

        return r;
    }

private:
    Vehicle::IPlateRecognize *detector;
    vector<Vehicle::PlateInfo> r;
};


BOOST_PYTHON_MODULE(_VehiclePlate)
{
    class_<cv::Rect>("zone",no_init)
            .add_property("x",&cv::Rect::x)
            .add_property("y",&cv::Rect::y)
            .add_property("w",&cv::Rect::width)
            .add_property("h",&cv::Rect::height);

    class_<Vehicle::PlateInfo>("PlateInfo",no_init)
            .add_property("zone",&Vehicle::PlateInfo::zone)
            .add_property("score",&Vehicle::PlateInfo::score)
            .add_property("license",&Vehicle::PlateInfo::license)
            .add_property("color",&Vehicle::PlateInfo::color);

    class_<vector<Vehicle::PlateInfo> >("PlateInfos",no_init)
            .def(vector_indexing_suite<vector<Vehicle::PlateInfo> >())
            .def("size",&vector<Vehicle::PlateInfo>::size);

    class_<PlateDetector>("PlateDetector",init<const string&,const int>())
            .def("detect",&PlateDetector::detect);
}
