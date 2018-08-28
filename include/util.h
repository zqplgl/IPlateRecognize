//
// Created by  on 04/04/2017.
//
#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
enum NiblackVersion
{
	NIBLACK = 0,
	SAUVOLA,
	WOLFJOLION,
};
#define BINARIZEWOLF_DEFAULTDR 128.0
#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;
namespace util{

    template <class T> void swap ( T& a, T& b )
    {
        T c(a); a=b; b=c;
    }

    template <class T> T min(T& a,T& b )
    {
        return a>b?b:a;

    }

	inline cv::Mat cropFromImage(const cv::Mat &image,cv::Rect rect){
        int w = image.cols-1;
        int h = image.rows-1;
        rect.x = std::max(rect.x,0);
        rect.y = std::max(rect.y,0);
        rect.height = std::min(rect.height,h-rect.y);
        rect.width = std::min(rect.width,w-rect.x);
        cv::Mat temp(rect.size(), image.type());
        cv::Mat cropped;
        temp = image(rect);
        temp.copyTo(cropped);
        return cropped;

    }

	inline cv::Mat cropBox2dFromImage(const cv::Mat &image,cv::RotatedRect rect)
    {
        cv::Mat M, rotated, cropped;
        float angle = rect.angle;
        cv::Size rect_size(rect.size.width,rect.size.height);
        if (rect.angle < -45.) {
            angle += 90.0;
            swap(rect_size.width, rect_size.height);
        }
        M = cv::getRotationMatrix2D(rect.center, angle, 1.0);
        cv::warpAffine(image, rotated, M, image.size(), cv::INTER_CUBIC);
        cv::getRectSubPix(rotated, rect_size, rect.center, cropped);
        return cropped;
    }

	inline cv::Mat calcHist(const cv::Mat &image)
    {
        cv::Mat hsv;
        std::vector<cv::Mat> hsv_planes;
        cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
        cv::split(hsv,hsv_planes);
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0,255};
        const float* histRange = {range};

        cv::calcHist( &hsv_planes[0], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange,true, true);
        return hist;

    }
	inline float computeSimilir(const cv::Mat &A,const cv::Mat &B)
    {

        cv::Mat histA,histB;
        histA = calcHist(A);
        histB = calcHist(B);
        return cv::compareHist(histA,histB,0);

    }


	inline double calcLocalStats(Mat &im, Mat &map_m, Mat &map_s, int winx, int winy) {
		Mat im_sum, im_sum_sq;
		cv::integral(im, im_sum, im_sum_sq, CV_64F);

		double m, s, max_s, sum, sum_sq;
		int wxh = winx / 2;
		int wyh = winy / 2;
		int x_firstth = wxh;
		int y_lastth = im.rows - wyh - 1;
		int y_firstth = wyh;
		double winarea = winx*winy;

		max_s = 0;
		for (int j = y_firstth; j <= y_lastth; j++) {
			sum = sum_sq = 0;

			sum = im_sum.at<double>(j - wyh + winy, winx) - im_sum.at<double>(j - wyh, winx) - im_sum.at<double>(j - wyh + winy, 0) + im_sum.at<double>(j - wyh, 0);
			sum_sq = im_sum_sq.at<double>(j - wyh + winy, winx) - im_sum_sq.at<double>(j - wyh, winx) - im_sum_sq.at<double>(j - wyh + winy, 0) + im_sum_sq.at<double>(j - wyh, 0);

			m = sum / winarea;
			s = sqrt((sum_sq - m*sum) / winarea);
			if (s > max_s) max_s = s;

			map_m.fset(x_firstth, j, m);
			map_s.fset(x_firstth, j, s);

			// Shift the window, add and remove	new/old values to the histogram
			for (int i = 1; i <= im.cols - winx; i++) {

				// Remove the left old column and add the right new column
				sum -= im_sum.at<double>(j - wyh + winy, i) - im_sum.at<double>(j - wyh, i) - im_sum.at<double>(j - wyh + winy, i - 1) + im_sum.at<double>(j - wyh, i - 1);
				sum += im_sum.at<double>(j - wyh + winy, i + winx) - im_sum.at<double>(j - wyh, i + winx) - im_sum.at<double>(j - wyh + winy, i + winx - 1) + im_sum.at<double>(j - wyh, i + winx - 1);

				sum_sq -= im_sum_sq.at<double>(j - wyh + winy, i) - im_sum_sq.at<double>(j - wyh, i) - im_sum_sq.at<double>(j - wyh + winy, i - 1) + im_sum_sq.at<double>(j - wyh, i - 1);
				sum_sq += im_sum_sq.at<double>(j - wyh + winy, i + winx) - im_sum_sq.at<double>(j - wyh, i + winx) - im_sum_sq.at<double>(j - wyh + winy, i + winx - 1) + im_sum_sq.at<double>(j - wyh, i + winx - 1);

				m = sum / winarea;
				s = sqrt((sum_sq - m*sum) / winarea);
				if (s > max_s) max_s = s;

				map_m.fset(i + wxh, j, m);
				map_s.fset(i + wxh, j, s);
			}
		}

		return max_s;
	}
	inline void NiblackSauvolaWolfJolion(Mat im, Mat output, NiblackVersion version, int winx, int winy, double k, double dR = BINARIZEWOLF_DEFAULTDR)
	{
		double m, s, max_s;
		double th = 0;
		double min_I, max_I;
		int wxh = winx / 2;
		int wyh = winy / 2;
		int x_firstth = wxh;
		int x_lastth = im.cols - wxh - 1;
		int y_lastth = im.rows - wyh - 1;
		int y_firstth = wyh;
		//int mx, my;

		// Create local statistics and store them in a double matrices
		Mat map_m = Mat::zeros(im.rows, im.cols, CV_32F);
		Mat map_s = Mat::zeros(im.rows, im.cols, CV_32F);
		max_s = calcLocalStats(im, map_m, map_s, winx, winy);

		minMaxLoc(im, &min_I, &max_I);

		Mat thsurf(im.rows, im.cols, CV_32F);

		// Create the threshold surface, including border processing
		// ----------------------------------------------------

		for (int j = y_firstth; j <= y_lastth; j++) {

			// NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
			for (int i = 0; i <= im.cols - winx; i++) {

				m = map_m.fget(i + wxh, j);
				s = map_s.fget(i + wxh, j);

				// Calculate the threshold
				switch (version) {

				case NIBLACK:
					th = m + k*s;
					break;

				case SAUVOLA:
					th = m * (1 + k*(s / dR - 1));
					break;

				case WOLFJOLION:
					th = m + k * (s / max_s - 1) * (m - min_I);
					break;

				default:
					cout << "Unknown threshold type in ImageThresholder::surfaceNiblackImproved()\n";
					exit(1);
				}

				thsurf.fset(i + wxh, j, th);

				if (i == 0) {
					// LEFT BORDER
					for (int i = 0; i <= x_firstth; ++i)
						thsurf.fset(i, j, th);

					// LEFT-UPPER CORNER
					if (j == y_firstth)
						for (int u = 0; u<y_firstth; ++u)
							for (int i = 0; i <= x_firstth; ++i)
								thsurf.fset(i, u, th);

					// LEFT-LOWER CORNER
					if (j == y_lastth)
						for (int u = y_lastth + 1; u<im.rows; ++u)
							for (int i = 0; i <= x_firstth; ++i)
								thsurf.fset(i, u, th);
				}

				// UPPER BORDER
				if (j == y_firstth)
					for (int u = 0; u<y_firstth; ++u)
						thsurf.fset(i + wxh, u, th);

				// LOWER BORDER
				if (j == y_lastth)
					for (int u = y_lastth + 1; u<im.rows; ++u)
						thsurf.fset(i + wxh, u, th);
			}

			// RIGHT BORDER
			for (int i = x_lastth; i<im.cols; ++i)
				thsurf.fset(i, j, th);

			// RIGHT-UPPER CORNER
			if (j == y_firstth)
				for (int u = 0; u<y_firstth; ++u)
					for (int i = x_lastth; i<im.cols; ++i)
						thsurf.fset(i, u, th);

			// RIGHT-LOWER CORNER
			if (j == y_lastth)
				for (int u = y_lastth + 1; u<im.rows; ++u)
					for (int i = x_lastth; i<im.cols; ++i)
						thsurf.fset(i, u, th);
		}

		for (int y = 0; y<im.rows; ++y)
			for (int x = 0; x<im.cols; ++x)
			{
				if (im.uget(x, y) >= thsurf.fget(x, y))
				{
					output.uset(x, y, 255);
				}
				else
				{
					output.uset(x, y, 0);
				}
			}
	}
	inline void clearLiuDingOnlyWhite(Mat &img) {
		const int x1 = 12;
		const int x2 = 12;
		Mat jump1 = Mat::zeros(1, img.rows, CV_32F);
		Mat jump2 = Mat::zeros(1, img.rows, CV_32F);
		for (int i = 0; i < img.rows / 3; i++) {
			int jumpCount = 0;
			int whiteCount = 0;
			for (int j = 0; j < img.cols - 1; j++) {
				if (img.at<char>(i, j) != img.at<char>(i, j + 1)) jumpCount++;

				if (img.at<uchar>(i, j) == 255) {
					whiteCount++;
				}
			}

			jump1.at<float>(i) = (float)jumpCount;
		}
		for (int i = img.rows - 1; i > 2 * img.rows / 3; i--) {
			int jumpCount = 0;
			int whiteCount = 0;
			for (int j = 0; j < img.cols - 1; j++) {
				if (img.at<char>(i, j) != img.at<char>(i, j + 1)) jumpCount++;

				if (img.at<uchar>(i, j) == 255) {
					whiteCount++;
				}
			}

			jump2.at<float>(i) = (float)jumpCount;
		}
		for (int i = 0; i < img.rows / 3; i++) {
			if (jump1.at<float>(i) <= x1) {
				for (int j = 0; j < img.cols; j++) {
					img.at<char>(i, j) = 0;
				}
			}
		}
		for (int i = img.rows - 1; i >2 * img.rows / 3; i--) {
			if (jump2.at<float>(i) <= x2) {
				for (int j = 0; j < img.cols; j++) {
					img.at<char>(i, j) = 0;
				}
			}
		}

	}
	inline void clearLiuDingOnlyWhite2(Mat &img) {

		const int x2 = 5;

		Mat jump2 = Mat::zeros(1, img.rows, CV_32F);

		for (int i = img.rows - 1; i > 2 * img.rows / 3; i--) {
			int jumpCount = 0;
			int whiteCount = 0;
			for (int j = 0; j < img.cols - 1; j++) {
				if (img.at<char>(i, j) != img.at<char>(i, j + 1)) jumpCount++;

				if (img.at<uchar>(i, j) == 255) {
					whiteCount++;
				}
			}

			jump2.at<float>(i) = (float)jumpCount;
		}
		for (int i = 0; i < img.rows / 3; i++) {

			for (int j = 0; j < img.cols; j++) {
				img.at<char>(i, j) = 0;
			}

		}
		for (int i = img.rows - 1; i >2 * img.rows / 3; i--) {
			if (jump2.at<float>(i) <= x2) {
				for (int j = 0; j < img.cols; j++) {
					img.at<char>(i, j) = 0;
				}
			}
		}

	}
	inline void clearLiuDingOnlyYellow2(Mat &img) {

		const int x2 = 5;

		Mat jump2 = Mat::zeros(1, img.rows, CV_32F);

		for (int i = img.rows - 1; i > 2 * img.rows / 3; i--) {
			int jumpCount = 0;
			int whiteCount = 0;
			for (int j = 0; j < img.cols - 1; j++) {
				if (img.at<char>(i, j) != img.at<char>(i, j + 1)) jumpCount++;

				if (img.at<uchar>(i, j) == 255) {
					whiteCount++;
				}
			}

			jump2.at<float>(i) = (float)jumpCount;
		}
		for (int i = 0; i < img.rows / 3; i++) {

			for (int j = 0; j < img.cols; j++) {
				img.at<char>(i, j) = 0;
			}

		}
		for (int i = img.rows - 1; i >2 * img.rows / 3; i--) {
			if (jump2.at<float>(i) <= x2) {
				for (int j = 0; j < img.cols; j++) {
					img.at<char>(i, j) = 0;
				}
			}
		}

	}

	inline int cut2LayerPlatePositionYellow2(Mat plateMat)
	{

		Mat ProjectMat = Mat::zeros(plateMat.cols, plateMat.rows, CV_8UC1);
		vector<int > nonzero;
		int countpixel = 0;
		for (int j = 0; j < plateMat.rows; j++)
		{
			for (int k = 0; k < plateMat.cols; k++)
			{
				if ((int)(*(plateMat.data + plateMat.step[0] * j + plateMat.step[1] * k))>0)
					countpixel++;
			}
			nonzero.push_back(countpixel);
			countpixel = 0;
		}

		for (int k = 0; k < ProjectMat.cols; k++)
		{
			if (nonzero[k] <= 2)continue;
			for (int j = ProjectMat.rows - 1; j >(ProjectMat.rows - 1 - nonzero[k]); j--)//ProjectMat.rows - nonzero[k]
			{
				*(ProjectMat.data + ProjectMat.step[0] * j + ProjectMat.step[1] * k) = 255;//widthcharacter
			}
		}
		vector<int >partvalue;
		vector<int >partx;
		for (int i = 15; i < nonzero.size() - 15; i++)
		{
			partvalue.push_back(nonzero[i]);
			partx.push_back(i);
		}
		//find the min value
		int minvalue = partvalue[29];
		int minx = partx[29];
		for (int i = partvalue.size() - 1; i >1; i--)
		{
			if (partvalue[i] < minvalue)
			{
				minvalue = partvalue[i];
				minx = partx[i];
			}
		}
		return minx;
	}

	inline int cut2LayerPlatePositionWhite2(Mat plateMat)
	{

		Mat ProjectMat = Mat::zeros(plateMat.cols, plateMat.rows, CV_8UC1);
		vector<int > nonzero;
		int countpixel = 0;
		for (int j = 0; j < plateMat.rows; j++)
		{
			for (int k = 0; k < plateMat.cols; k++)
			{
				if ((int)(*(plateMat.data + plateMat.step[0] * j + plateMat.step[1] * k))>0)
					countpixel++;
			}
			nonzero.push_back(countpixel);
			countpixel = 0;
		}

		for (int k = 0; k < ProjectMat.cols; k++)
		{
			if (nonzero[k] <= 2)continue;
			for (int j = ProjectMat.rows - 1; j >(ProjectMat.rows - 1 - nonzero[k]); j--)//ProjectMat.rows - nonzero[k]
			{
				*(ProjectMat.data + ProjectMat.step[0] * j + ProjectMat.step[1] * k) = 255;//widthcharacter
			}
		}
		vector<int >partvalue;
		vector<int >partx;
		for (int i = 15; i < nonzero.size() - 15; i++)
		{
			partvalue.push_back(nonzero[i]);
			partx.push_back(i);
		}
		//find the min value
		int minvalue = partvalue[29];
		int minx = partx[29];
		for (int i = partvalue.size() - 1; i >1; i--)
		{
			if (partvalue[i] < minvalue)
			{
				minvalue = partvalue[i];
				minx = partx[i];
			}
		}
		return minx;
	}
}//namespace util
