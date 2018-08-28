// testDllPlateDtection2015.cpp : 定义控制台应用程序的入口点。
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>  
#include <fstream>  
#include <iterator>  
#include <vector>  
#include "time.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "PlateInfo.h"
#include "IDllPlateDetection2015.h"


using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	//import the vehicle detection dll车辆检测模型
	string platemodelpath = "/home/zqp/install_lib/models";
	//cout << "it is beginning!" << endl;
	IPlateDetectionInterface *detectorPlate = CreateDetector(platemodelpath);
    string picname = "/home/zqp/testimage/image3/00BD50C6-C75A-4247-98B0-89AB44FE5955_325514_03_06.jpg";
	
	//get the path of image file
	vector<string> currentImage;
    currentImage.push_back(picname);
	int currentFileCount = currentImage.size();
	const char* imgFileName;

	for (int k = 0; k < currentFileCount; k++)
	{
		imgFileName = currentImage[k].c_str();
		std::cout << currentFileCount << "-" << k << imgFileName << endl;

		Mat inputImg = cv::imread(imgFileName, 1);
		const float confidence_threshold = 0.1;
		int plateClass;
		int x = 0, y = 0, w = 0, h = 0;
		PlateInfo plateresult = detectorPlate->PlateRecogPipeline(inputImg.data, inputImg.cols, inputImg.rows, confidence_threshold);
		string plateName = plateresult.getPlateName();
		float confidence = plateresult.confidence;
		cout << "confidence: " << confidence << endl;
		cout << "platename: " << plateName << endl;
	}

    return 0;
}

