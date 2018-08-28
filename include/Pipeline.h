//
// Created by  on 22/10/2017.
//

#ifndef SWIFTPR_PIPLINE_H
#define SWIFTPR_PIPLINE_H

#include "PlateDetection.h"
#include "PlateInfo.h"
#include "FastDeskew.h"
#include "FineMapping.h"
#include "SegmentationFreeRecognizer.h"

namespace pr{

    const std::vector<std::string> CH_PLATE_CODE{"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂","琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z","港","学","使","警","澳","挂","军","北","南","广","沈","兰","成","济","海","民","航","空"};
	
    class PipelinePR{
        public:
           
            PlateDetection *plateDetection;
            FineMapping *fineMapping;
            SegmentationFreeRecognizer *segmentationFreeRecognizer;
			PipelinePR() {};
			void initial(std::string detector_filename,
                       std::string finemapping_prototxt,std::string finemapping_caffemodel,
                       std::string segmentationfree_proto,std::string segmentationfree_caffemodel

                       )
			{
				plateDetection = new PlateDetection(detector_filename);
				fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
				segmentationFreeRecognizer = new SegmentationFreeRecognizer(segmentationfree_proto, segmentationfree_caffemodel);
			
			};
			~PipelinePR() {};
		
            std::vector<std::string> plateRes;
     };


}
#endif //SWIFTPR_PIPLINE_H
