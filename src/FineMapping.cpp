//
// Created by  on 22/09/2017.
//

#include "FineMapping.h"
#include "util.h"

using namespace cv;
using namespace std;
using namespace util;
namespace pr{

    const int FINEMAPPING_H = 50;
    const int FINEMAPPING_W = 120;
    const int PADDING_UP_DOWN = 30;
    void drawRect(cv::Mat image,cv::Rect rect)
    {
        cv::Point p1(rect.x,rect.y);
        cv::Point p2(rect.x+rect.width,rect.y+rect.height);
        cv::rectangle(image,p1,p2,cv::Scalar(0,255,0),1);
    }


    FineMapping::FineMapping(std::string prototxt,std::string caffemodel) {
         net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);

    }

    cv::Mat FineMapping::FineMappingHorizon(cv::Mat FinedVertical,int leftPadding,int rightPadding)
    {

		cv::Mat inputBlob = cv::dnn::blobFromImage(FinedVertical, 1 / 255.0, cv::Size(66, 16),cv::Scalar(0,0,0),false);

        net.setInput(inputBlob,"data");
        cv::Mat prob = net.forward();
        int front = static_cast<int>(prob.at<float>(0,0)*FinedVertical.cols);
        int back = static_cast<int>(prob.at<float>(0,1)*FinedVertical.cols);
        front -= leftPadding ;
        if(front<0) front = 0;
        back +=rightPadding;
        if(back>FinedVertical.cols-1) back=FinedVertical.cols - 1;
        cv::Mat cropped  = FinedVertical.colRange(front,back).clone();
		return cropped;
    }
    std::pair<int,int> FitLineRansac(std::vector<cv::Point> pts,int zeroadd = 0 )
    {
        std::pair<int,int> res;
        if(pts.size()>2)
        {
            cv::Vec4f line;
            cv::fitLine(pts,line,cv::DIST_HUBER,0,0.01,0.01);
            float vx = line[0];
            float vy = line[1];
            float x = line[2];
            float y = line[3];
            int lefty = static_cast<int>((-x * vy / vx) + y);
            int righty = static_cast<int>(((136- x) * vy / vx) + y);
            res.first = lefty+PADDING_UP_DOWN+zeroadd;
            res.second = righty+PADDING_UP_DOWN+zeroadd;
            return res;
        }
        res.first = zeroadd;
        res.second = zeroadd;
        return res;
    }

    cv::Mat FineMapping::FineMappingVertical(cv::Mat InputProposal, int plateClass, int sliceNum,int upper,int lower,int windows_size)
    {
        cv::Mat PreInputProposal;
        cv::Mat proposal;

        cv::resize(InputProposal,PreInputProposal,cv::Size(FINEMAPPING_W,FINEMAPPING_H));
		if (InputProposal.channels() == 3)
		{
			cv::cvtColor(PreInputProposal, proposal, cv::COLOR_BGR2GRAY);
			if(plateClass==2|| plateClass == 3|| plateClass == 5|| plateClass == 6|| plateClass == 4 || plateClass == 7)
			bitwise_not(proposal, proposal);  //yellow white new 
		}
		else
		{
			PreInputProposal.copyTo(proposal);
			if (plateClass == 2 || plateClass == 3 || plateClass == 5 || plateClass == 6 || plateClass == 4 || plateClass == 7)
			bitwise_not(proposal, proposal);  //yellow white new 
		}

        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(1,3));
		cv::Mat quad;
		Mat affine_mat;
		if (plateClass == 1||plateClass == 2 || plateClass == 3 || plateClass == 5 || plateClass == 6)
		{
			float diff = static_cast<float>(upper - lower);
			diff /= static_cast<float>(sliceNum - 1);
			cv::Mat binary_adaptive;
			std::vector<cv::Point> line_upper;
			std::vector<cv::Point> line_lower;
			int contours_nums = 0;

			for (int i = 0; i < sliceNum; i++)
			{
				std::vector<std::vector<cv::Point> > contours;
				float k = lower + i*diff;
				int win = 25;//200  nut 是1000
				Mat imageout = cv::Mat::zeros(proposal.size(), CV_8U);
				util::NiblackSauvolaWolfJolion(proposal, imageout, SAUVOLA, win, win, 0.0001* k);
				util::clearLiuDingOnlyWhite(imageout);

				cv::Mat draw;
				imageout.copyTo(draw);
				cv::findContours(imageout, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				for (auto contour : contours)
				{
					cv::Rect bdbox = cv::boundingRect(contour);
					float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
					int  bdboxAera = bdbox.width*bdbox.height;
					if ((lwRatio > 0.6&&bdbox.width*bdbox.height > 50 && bdboxAera < 500)
						|| (lwRatio > 3.0 && bdboxAera < 110 && bdboxAera>10))
					{

						cv::Point p1(bdbox.x, bdbox.y);
						cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
						line_upper.push_back(p1);
						line_lower.push_back(p2);
						contours_nums += 1;
					}
				}
			}

			if (contours_nums < 50)
			{
				cv::bitwise_not(InputProposal, InputProposal);
				cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 5));
				cv::Mat bak;
				cv::resize(InputProposal, bak, cv::Size(FINEMAPPING_W, FINEMAPPING_H));
				cv::erode(bak, bak, kernal);//erode
				if (InputProposal.channels() == 3)
				{
					cv::cvtColor(bak, proposal, cv::COLOR_BGR2GRAY);
					//bitwise_not(proposal, proposal);  //yellow white new 
				}
				else
				{
					proposal = bak;
				}
				int contours_nums = 0;

				for (int i = 0; i < sliceNum; i++)
				{
					std::vector<std::vector<cv::Point> > contours;
					float k = lower + i*diff;

					//定义参数进行二值化
					//int kk = 2;   //1  nut 是2
					int win = 15;//200  nut 是1000
					Mat imageout = cv::Mat::zeros(proposal.size(), CV_8U);
					util::NiblackSauvolaWolfJolion(proposal, imageout, NIBLACK, win, win, 0.001* k);//0.18SAUVOLA
					util::clearLiuDingOnlyWhite(imageout);
					//              cv::imshow("image",binary_adaptive);
					//              cv::waitKey(0);
					cv::Mat draw;
					//binary_adaptive.copyTo(draw);
					//cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

					imageout.copyTo(draw);
					cv::findContours(imageout, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
					for (auto contour : contours)
					{
						cv::Rect bdbox = cv::boundingRect(contour);
						float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
						int  bdboxAera = bdbox.width*bdbox.height;
						if ((lwRatio > 0.7&&bdbox.width*bdbox.height > 120 && bdboxAera < 400)//300
							|| (lwRatio > 3.0 && bdboxAera < 100 && bdboxAera>10))
						{

							cv::Point p1(bdbox.x, bdbox.y);
							cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
							line_upper.push_back(p1);
							line_lower.push_back(p2);
							contours_nums += 1;
						}
					}
				}
			}
			cv::Mat rgb;
			cv::copyMakeBorder(PreInputProposal, rgb, 30, 30, 0, 0, cv::BORDER_REPLICATE);

			std::pair<int, int> A;
			std::pair<int, int> B;
			A = FitLineRansac(line_upper, -2);
			B = FitLineRansac(line_lower, 2);
			int leftyB = A.first;
			int rightyB = A.second;
			int leftyA = B.first;
			int rightyA = B.second;
			int cols = rgb.cols;
			int rows = rgb.rows;

			std::vector<cv::Point2f> corners(4);//4
			corners[0] = cv::Point2f(cols - 1, rightyA + 2);
			corners[1] = cv::Point2f(0, leftyA + 2);
			corners[2] = cv::Point2f(cols - 1, rightyB - 2);
			corners[3] = cv::Point2f(0, leftyB - 2);

			std::vector<cv::Point2f> corners_trans(4);
			corners_trans[0] = cv::Point2f(136, 36);//36
			corners_trans[1] = cv::Point2f(0, 36);//36
			corners_trans[2] = cv::Point2f(136, 0);
			corners_trans[3] = cv::Point2f(0, 0);


			cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
			quad = cv::Mat::zeros(36, 136, CV_8UC3);//36
			cv::warpPerspective(rgb, quad, transform, quad.size());
		}
		else if (plateClass == 4 || plateClass == 7)
		{
			float diff = static_cast<float>(upper - lower);
			diff /= static_cast<float>(sliceNum - 1);
			cv::Mat binary_adaptive;
			std::vector<cv::Point> line_upper;
			std::vector<cv::Point> line_lower;
			int contours_nums = 0;

			for (int i = 0; i < sliceNum; i++)
			{
				std::vector<std::vector<cv::Point> > contours;
				float k = lower + i*diff;
				cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);

				util::clearLiuDingOnlyWhite2(binary_adaptive);
				cv::Mat draw;
				binary_adaptive.copyTo(draw);
				cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
				for (auto contour : contours)
				{
					cv::Rect bdbox = cv::boundingRect(contour);
					float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
					int  bdboxAera = bdbox.width*bdbox.height;
					if ((lwRatio > 0.7&&bdbox.width*bdbox.height > 120 && bdboxAera < 400)
						|| (lwRatio > 3.0 && bdboxAera < 110 && bdboxAera>10&& bdbox.x>4&& bdbox.x<(binary_adaptive.cols-4)))
					{

						cv::Point p1(bdbox.x, bdbox.y);
						cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
						line_upper.push_back(p1);
						line_lower.push_back(p2);
						contours_nums += 1;
					}
				}
			}

			if (contours_nums < 20)
			{
				cv::bitwise_not(InputProposal, InputProposal);
				cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 3));
				cv::Mat bak;
				cv::resize(InputProposal, bak, cv::Size(FINEMAPPING_W, FINEMAPPING_H));
				//cv::erode(bak, bak, kernal);//erode
				if (InputProposal.channels() == 3)
				{
					cv::cvtColor(bak, proposal, cv::COLOR_BGR2GRAY);
					//bitwise_not(proposal, proposal);  //yellow white new 
				}
				else
				{
					proposal = bak;
				}
				int contours_nums = 0;

				for (int i = 0; i < sliceNum; i++)
				{
					std::vector<std::vector<cv::Point> > contours;
					float k = lower + i*diff;
					cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);
					//clearLiuDingOnlyWhite2(binary_adaptive);
					//定义参数进行二值化
					//int kk = 2;   //1  nut 是2
					//int win = 15;//200  nut 是1000
					//Mat imageout = cv::Mat::zeros(proposal.size(), CV_8U);
					//NiblackSauvolaWolfJolion(proposal, imageout, NIBLACK, win, win, 0.001* k);//0.18SAUVOLA
					//clearLiuDingOnlyWhite(imageout);
					//              cv::imshow("image",binary_adaptive);
					//              cv::waitKey(0);
					cv::Mat draw;
					binary_adaptive.copyTo(draw);
					cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

					//imageout.copyTo(draw);
					//cv::findContours(imageout, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
					for (auto contour : contours)
					{
						cv::Rect bdbox = cv::boundingRect(contour);
						//if (bdbox.x < 4)continue;
						float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
						int  bdboxAera = bdbox.width*bdbox.height;
						if ((lwRatio > 0.5&&bdbox.width*bdbox.height > 80 && bdboxAera < 500 )//300
							|| (lwRatio > 3.0 && bdboxAera < 120 && bdboxAera>10))
						{

							cv::Point p1(bdbox.x, bdbox.y);
							cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
							line_upper.push_back(p1);
							line_lower.push_back(p2);
							contours_nums += 1;
						}
					}
				}
			}
#if 0
			for (int i = 0;i < line_upper.size();i++)
			{
				cv::circle(PreInputProposal, line_upper[i], 1, cv::Scalar(0, 255, 0), 1);
			}
			for (int i = 0;i < line_lower.size();i++)
			{
				cv::circle(PreInputProposal, line_lower[i], 1, cv::Scalar(0, 0, 255), 1);
			}
#endif
			cv::Mat rgb;
			cv::copyMakeBorder(PreInputProposal, rgb, 30, 30, 0, 0, cv::BORDER_REPLICATE);
			//cv::imshow("rgb",rgb);
			//cv::waitKey(0);
			//

			std::pair<int, int> A;
			std::pair<int, int> B;
			A = FitLineRansac(line_upper, -2);
			B = FitLineRansac(line_lower, 2);
			int leftyB = A.first;
			int rightyB = A.second;
			int leftyA = B.first;
			int rightyA = B.second;
			int cols = rgb.cols;
			int rows = rgb.rows;

			//        pts_map1  = np.float32([[cols - 1, rightyA], [0, leftyA],[cols - 1, rightyB], [0, leftyB]])
			//        pts_map2 = np.float32([[136,36],[0,36],[136,0],[0,0]])
			//        mat = cv2.getPerspectiveTransform(pts_map1,pts_map2)
			//        image = cv2.warpPerspective(rgb,mat,(136,36),flags=cv2.INTER_CUBIC)
#if 1
			int upCut = rightyA - rightyB;
			std::vector<cv::Point2f> corners(4);
			corners[0] = cv::Point2f(cols - 1, rightyA);
			corners[1] = cv::Point2f(0, leftyA);
			if (plateClass == 7)
			{
				corners[2] = cv::Point2f(cols - 1, rightyB - upCut + 5);
				corners[3] = cv::Point2f(0, leftyB - upCut + 5);
			}
			if (plateClass == 4)
			{
				corners[2] = cv::Point2f(cols - 1, rightyB - upCut + 7);
				corners[3] = cv::Point2f(0, leftyB - upCut + 7);
			}
			std::vector<cv::Point2f> corners_trans(4);
			corners_trans[0] = cv::Point2f(136, 60);//36
			corners_trans[1] = cv::Point2f(0, 60);//36
			corners_trans[2] = cv::Point2f(136, 0);
			corners_trans[3] = cv::Point2f(0, 0);

			cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
			quad = cv::Mat::zeros(60, 136, CV_8UC3);//36
			cv::warpPerspective(rgb, quad, transform, quad.size());
#endif
#if 0
			cv::Mat quadBinary;
			cv::Mat quadgrey;
			cv::cvtColor(quad, quadgrey, cv::COLOR_BGR2GRAY);
			cv::adaptiveThreshold(quadgrey, quadBinary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 17, 5);

			Point2f dstTri[3];
			Point2f plTri[3];
			plTri[0] = Point2f(0 , 0);
			plTri[1] = Point2f(quad.cols - 7 - 1, 0);
			plTri[2] = Point2f(0+7, quad.rows - 1);

			dstTri[0] = Point2f(3.5, 0);
			dstTri[1] = Point2f(quad.cols - 1 - 3, 0);
			dstTri[2] = Point2f(3, quad.rows - 1);

			Mat warp_mat = getAffineTransform(plTri, dstTri);
			
			affine_mat.create((int)quad.rows, (int)quad.cols, CV_8UC3);
			warpAffine(quad, affine_mat, warp_mat, affine_mat.size(), CV_INTER_AREA);
#endif
		}
		return quad; //affine_mat;/

    }

	cv::Mat FineMapping::FineMappingVertical2(cv::Mat InputProposal, int sliceNum, int upper, int lower, int windows_size) {


		cv::Mat PreInputProposal;
		cv::Mat proposal;

		cv::resize(InputProposal, PreInputProposal, cv::Size(FINEMAPPING_W, FINEMAPPING_H));
		if (InputProposal.channels() == 3)
		{
			cv::cvtColor(PreInputProposal, proposal, cv::COLOR_BGR2GRAY);
			bitwise_not(proposal, proposal);  //yellow white new 
		}
		else
			PreInputProposal.copyTo(proposal);

		//            proposal = PreInputProposal;

		// this will improve some sen
		cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 3));
		//        cv::erode(proposal,proposal,kernal);


		float diff = static_cast<float>(upper - lower);
		diff /= static_cast<float>(sliceNum - 1);
		cv::Mat binary_adaptive;
		std::vector<cv::Point> line_upper;
		std::vector<cv::Point> line_lower;
		int contours_nums = 0;

		for (int i = 0; i < sliceNum; i++)
		{
			std::vector<std::vector<cv::Point> > contours;
			float k = lower + i*diff;
			cv::adaptiveThreshold(proposal, binary_adaptive, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, windows_size, k);
			cv::Mat draw;
			binary_adaptive.copyTo(draw);
			cv::findContours(binary_adaptive, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			for (auto contour : contours)
			{
				cv::Rect bdbox = cv::boundingRect(contour);
				float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
				int  bdboxAera = bdbox.width*bdbox.height;
				if ((lwRatio>0.7&&bdbox.width*bdbox.height>100 && bdboxAera<300)
					|| (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
				{

					cv::Point p1(bdbox.x, bdbox.y);
					cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
					line_upper.push_back(p1);
					line_lower.push_back(p2);
					contours_nums += 1;
				}
			}
		}

		if (contours_nums<41)
		{
			cv::bitwise_not(InputProposal, InputProposal);
			cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 5));
			cv::Mat bak;
			cv::resize(InputProposal, bak, cv::Size(FINEMAPPING_W, FINEMAPPING_H));
			cv::erode(bak, bak, kernal);
			if (InputProposal.channels() == 3)
				cv::cvtColor(bak, proposal, cv::COLOR_BGR2GRAY);
			else
				proposal = bak;
			int contours_nums = 0;

			for (int i = 0; i < sliceNum; i++)
			{
				std::vector<std::vector<cv::Point> > contours;
				float k = lower + i*diff;
				cv::adaptiveThreshold(proposal, binary_adaptive, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, windows_size, k);
				//                cv::imshow("image",binary_adaptive);
				//            cv::waitKey(0);
				cv::Mat draw;
				binary_adaptive.copyTo(draw);
				cv::findContours(binary_adaptive, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				for (auto contour : contours)
				{
					cv::Rect bdbox = cv::boundingRect(contour);
					float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
					int  bdboxAera = bdbox.width*bdbox.height;
					if ((lwRatio>0.7&&bdbox.width*bdbox.height>120 && bdboxAera<300)
						|| (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
					{

						cv::Point p1(bdbox.x, bdbox.y);
						cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
						line_upper.push_back(p1);
						line_lower.push_back(p2);
						contours_nums += 1;
					}
				}
			}
		}

		cv::Mat rgb;
		cv::copyMakeBorder(PreInputProposal, rgb, 30, 30, 0, 0, cv::BORDER_REPLICATE);

		std::pair<int, int> A;
		std::pair<int, int> B;
		A = FitLineRansac(line_upper, -2);
		B = FitLineRansac(line_lower, 2);
		int leftyB = A.first;
		int rightyB = A.second;
		int leftyA = B.first;
		int rightyA = B.second;
		int cols = rgb.cols;
		int rows = rgb.rows;
		//        pts_map1  = np.float32([[cols - 1, rightyA], [0, leftyA],[cols - 1, rightyB], [0, leftyB]])
		//        pts_map2 = np.float32([[136,36],[0,36],[136,0],[0,0]])
		//        mat = cv2.getPerspectiveTransform(pts_map1,pts_map2)
		//        image = cv2.warpPerspective(rgb,mat,(136,36),flags=cv2.INTER_CUBIC)
		std::vector<cv::Point2f> corners(4);
		corners[0] = cv::Point2f(cols - 1, rightyA);
		corners[1] = cv::Point2f(0, leftyA);
		corners[2] = cv::Point2f(cols - 1, rightyB);
		corners[3] = cv::Point2f(0, leftyB);
		std::vector<cv::Point2f> corners_trans(4);
		corners_trans[0] = cv::Point2f(136, 36);
		corners_trans[1] = cv::Point2f(0, 36);
		corners_trans[2] = cv::Point2f(136, 0);
		corners_trans[3] = cv::Point2f(0, 0);
		cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
		cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);
		cv::warpPerspective(rgb, quad, transform, quad.size());
		return quad;

	}

}


