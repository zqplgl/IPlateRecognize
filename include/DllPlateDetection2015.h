#pragma once
#ifndef DLLPLATEDETECTION2015_H
#define DLLPLATEDETECTION2015_H

#include <caffe/caffe.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <IDllPlateDetection2015.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Pipeline.h"
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace pr;

char* CLASSES[8] = { "__background__","plateblue1", "platene1" ,"plateback1","platey2","platey1","platew1","platew2"};

class  Detector :public IPlateDetectionInterface
{
public:
	Detector(const string& model_file,
		const string& weights_file,
		const string& mean_file,
		const string& mean_value,
		const float normalize_value);

	~Detector();
	
	std::vector<vector<float> > Detect(const cv::Mat& img);
	virtual std::pair<std::string, float> PlateRecogniser(unsigned char* imagedata, int width, int height, int plateClass,int &whitePlateType);
	virtual  cv::Mat DetectPlate(unsigned char* imagedata, int width, int height,int &x, int &y, int &w, int &h, float confidence_threshold,int &plateClass);
	virtual PlateInfo PlateRecogPipeline(unsigned char* imagedata, int width, int height, float confidence_threshold);
private:
	void SetMean(const string& mean_file, const string& mean_value);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	
	void Preprocess(const cv::Mat& img,	std::vector<cv::Mat>* input_channels);
	
	void Preprocess(const cv::Mat& img,	std::vector<cv::Mat>* input_channels, double normalize_value);
	
private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	float nor_val = 1.0;
};

IPlateDetectionInterface *CreateDetector(const string& model_path);
#endif //CAFFE_CLASSIFIER_H
