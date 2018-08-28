#ifndef IPLATEDETECTIONINTERFACE_H
#define IPLATEDETECTIONINTERFACE_H

/**********************************************/
/*	@file ./IPlateDetectionInterface.h
*
*	@author liu
*
*	@date 2018-07-12
*
*	@version 1.0
*/
/**********************************************/

#include "PlateInfo.h"

using namespace std;
using namespace pr;

class  IPlateDetectionInterface
{
public:
	virtual cv::Mat DetectPlate(unsigned char* imagedata, int width, int height,int &x, int &y, int &w, int &h, float confidence_threshold, int &plateClass)=0;
	virtual std::pair<std::string, float> PlateRecogniser(unsigned char* imagedata, int width, int height, int plateClass,int &whitePlateType) = 0;
	virtual PlateInfo PlateRecogPipeline(unsigned char* imagedata, int width, int height, float confidence_threshold) = 0;
	virtual ~IPlateDetectionInterface() {}
	
};

IPlateDetectionInterface *CreateDetector(const string& model_path);

void DestroyDetector(IPlateDetectionInterface* Recognizer1);

#endif
