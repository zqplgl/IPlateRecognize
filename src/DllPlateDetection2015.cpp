#include "stdlib.h"
#include <string.h>
#include <stdio.h>
#include "DllPlateDetection2015.h"

#include "util.h"
#include "Iclassify.h"


pr::PipelinePR prc;
ClassifierInterface *platewhite;
const int HorizontalPadding = 4;

void decryptfile(char *in_filename, char *pwd, char *out_file)
{
	FILE *fp1, *fp2;
	register char ch;
	int j = 0;
	int j0 = 0;
	fp1 = fopen(in_filename, "r");
	if (fp1 == NULL) {
		printf("cannot open in-file./n");
		exit(1);
	}
	fp2 = fopen(out_file, "w");
	if (fp2 == NULL) {
		printf("cannot open or create out-file./n");
		exit(1);
	}

	while (pwd[++j0]);
	ch = fgetc(fp1);

	while (!feof(fp1)) {
		ch = ch - 23;
		fputc(ch, fp2);//
		ch = fgetc(fp1);
	}
	fclose(fp1);
	fclose(fp2);
}
Detector::Detector(const string& model_file,
	const string& weights_file,
	const string& mean_file,
	const string& mean_value,
	//const float confidence_threshold,
	const float normalize_value) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(weights_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file, mean_value);
	nor_val = normalize_value;
}

float sec(clock_t clocks)
{
	return (float)clocks / CLOCKS_PER_SEC;
}
std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	if (nor_val != 1.0) {
		Preprocess(img, &input_channels, nor_val);
	}
	else {
		Preprocess(img, &input_channels);
	}
	clock_t time;
	time = clock();
	net_->Forward();
	/* Copy the output layer to a std::vector */
	Blob<float>* result_blob = net_->output_blobs()[0];
	const float* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	vector<vector<float> > detections;
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {
			// Skip invalid detection.
			result += 7;
			continue;
		}
		vector<float> detection(result, result + 7);
		detections.push_back(detection);
		result += 7;
	}
	return detections;
}
inline int judgeCharRange(int id)
{return id<31 || id>64;}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
	cv::Scalar channel_mean;
	if (!mean_file.empty()) {
		CHECK(mean_value.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

		/* Convert from BlobProto to Blob<float> */
		Blob<float> mean_blob;
		mean_blob.FromProto(blob_proto);
		CHECK_EQ(mean_blob.channels(), num_channels_)
			<< "Number of channels of mean file doesn't match input layer.";

		/* The format of the mean file is planar 32-bit float BGR or grayscale. */
		std::vector<cv::Mat> channels;
		float* data = mean_blob.mutable_cpu_data();
		for (int i = 0; i < num_channels_; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
			channels.push_back(channel);
			data += mean_blob.height() * mean_blob.width();
		}

		/* Merge the separate channels into a single image. */
		cv::Mat mean;
		cv::merge(channels, mean);

		/* Compute the global mean pixel value and create a mean image
		* filled with this value. */
		channel_mean = cv::mean(mean);
		mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
	}
	if (!mean_value.empty()) {
		CHECK(mean_file.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
		stringstream ss(mean_value);
		vector<float> values;
		string item;
		while (getline(ss, item, ',')) {
			float value = std::atof(item.c_str());
			values.push_back(value);
		}
		CHECK(values.size() == 1 || values.size() == num_channels_) <<
			"Specify either 1 mean_value or as many as channels: " << num_channels_;

		std::vector<cv::Mat> channels;
		for (int i = 0; i < num_channels_; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
				cv::Scalar(values[i]));
			channels.push_back(channel);
		}
		cv::merge(channels, mean_);
	}
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Detector::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

void Detector::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels, double normalize_value) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3, normalize_value);
	else
		sample_resized.convertTo(sample_float, CV_32FC1, normalize_value);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

Mat Detector::DetectPlate(unsigned char* imagedata, int width, int height, int &x,int &y,int &w,int &h,float confidence_threshold, int &plateClass)
{
	Mat img(height, width, CV_8UC3, imagedata);

	cv::Mat plateMat;
	std::vector<vector<float> > detections = Detect(img);
	if(detections.size()>0&& detections[0][2]>0.0)//NO plate score is -1
	{
		const vector<float>& d = detections[0];
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold) {

			cv::Point pt1, pt2;
			pt1.x = (img.cols*d[3]);
			pt1.y = (img.rows*d[4]);
			pt2.x = (img.cols*d[5]);
			pt2.y = (img.rows*d[6]);

			cv::Rect plateRect;
			plateRect.x = pt1.x;
			plateRect.y = pt1.y-5;
			plateRect.width = pt2.x - pt1.x;
			plateRect.height = pt2.y - pt1.y+10;
			if (plateRect.y < 0)plateRect.y = 0;
			if (plateRect.x < 0)plateRect.x = 0;
			if (plateRect.y + plateRect.height > img.rows)plateRect.height = img.rows - plateRect.y;
			if (plateRect.x + plateRect.width > img.cols)plateRect.width = img.cols - plateRect.x;

			char label[100];
			plateClass = static_cast<int>(d[1]);
			sprintf(label, "%s,%f", CLASSES[static_cast<int>(d[1])], score);
			int baseline;
			cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
			cv::Point pt3;
			pt3.x = pt1.x + size.width;
			pt3.y = pt1.y - size.height;
			x = plateRect.x;
			y = plateRect.y;
			w = plateRect.width;
			h = plateRect.height;

			cv::Mat plateMat1(img, plateRect);
			plateMat1.copyTo(plateMat);
			
		}
	}
	else //cascade xml 
	{
		std::vector<PlateInfo> results;
		std::vector<pr::PlateInfo> plates;
		prc.plateDetection->plateDetectionRough(img, plates, 36, 700);
		cv::Mat image_finemapping;
		if(plates.size()>0)
		{
			image_finemapping = plates[0].getPlateImage();
			Rect plateRect = plates[0].getPlateRect();
			x = plateRect.x;
			y = plateRect.y;
			w = plateRect.width;
			h = plateRect.height;
			image_finemapping = prc.fineMapping->FineMappingVertical2(image_finemapping);
			image_finemapping = pr::fastdeskew(image_finemapping, 5);
			image_finemapping = prc.fineMapping->FineMappingHorizon(image_finemapping, 4, HorizontalPadding + 3);

		}
		image_finemapping.copyTo(plateMat);
		plateClass = 1;
	}
	return plateMat;
}

std::pair<std::string, float> Detector::PlateRecogniser(unsigned char* imagedata, int width, int height, int plateClass, int &whitePlateType)
{
	Mat img(height, width, CV_8UC3, imagedata);
	
	cv::Mat image_finemapping = prc.fineMapping->FineMappingVertical(img, plateClass);
	std::pair<vector<int>, float> res;
#if 1
	if (plateClass == 1 || plateClass == 2 || plateClass == 3 || plateClass == 5 || plateClass == 6)
	{
		if (plateClass == 6)
		{
			cv::Mat image_finemappingWhitePlate;
			cv::resize(image_finemapping, image_finemappingWhitePlate, cv::Size(136, 36));//
			cv::resize(image_finemapping, image_finemapping, cv::Size(140, 36));
			res = prc.segmentationFreeRecognizer->SegmentationFreeForSinglePlate(image_finemapping);

			Rect cutPlate;
			cutPlate.x = 3;
			cutPlate.y = 3;
			cutPlate.width = image_finemappingWhitePlate.cols-6;
			cutPlate.height = image_finemappingWhitePlate.rows - 6;;
			cv::Mat cutPlatemat(image_finemappingWhitePlate, cutPlate);
			whitePlateType = platewhite->classify(cutPlatemat);
			if (whitePlateType == 0)
			{
				vector<int> &plate = res.first;
				int ch = plate[0];
				if (!judgeCharRange(ch))
					whitePlateType = 1;//jundui palte is wrong,here change
				else
				{
					plate[plate.size() - 1] = 68;
				}

			}
			else if (whitePlateType == 1)//jundui
			{
			}
			else if (whitePlateType == 3)//wujing
			{
			}
		}
		else
		{
			Rect plateRect;
			plateRect.x = 2;
			plateRect.y = 2;
			plateRect.width = image_finemapping.cols-2;
			plateRect.height = image_finemapping.rows-4;

			Mat image_finemappingcut(image_finemapping, plateRect);

			cv::resize(image_finemappingcut, image_finemappingcut, cv::Size(140, 36));
			res = prc.segmentationFreeRecognizer->SegmentationFreeForSinglePlate(image_finemappingcut);
		}
	}
	//plateclass is Y2
	if (plateClass == 4)
	{
		cv::resize(image_finemapping, image_finemapping, cv::Size(140, 60));//image_finemapping
		cv::Mat proposal;
		if (image_finemapping.channels() == 3)
		{
			cv::cvtColor(image_finemapping, proposal, cv::COLOR_BGR2GRAY);
			bitwise_not(proposal, proposal);  //yellow white new 
		}
		else
		{
			image_finemapping.copyTo(proposal);
			bitwise_not(proposal, proposal);  //yellow white new 
		}
		//cut the double layer
		int k = 0.01, win = 22;
		Mat imageout = cv::Mat::zeros(image_finemapping.size(), CV_8U);
		util::NiblackSauvolaWolfJolion(proposal, imageout, SAUVOLA, win, win, 0.18 * k);//

		int cutPosition = util::cut2LayerPlatePositionYellow2(imageout);
		if (cutPosition > 35 || cutPosition < 15)cutPosition = 23;

		//cut to two parts
		Rect downPartRect,uppartRect;
		uppartRect.x = 0;
		uppartRect.y = 2;
		uppartRect.width = image_finemapping.cols;
		uppartRect.height = cutPosition;//;21;

		downPartRect.x = 0;
		downPartRect.y = cutPosition-2;
		downPartRect.width = image_finemapping.cols;
		downPartRect.height = image_finemapping.rows - cutPosition + 2;;

		cv::Mat uppartMat(image_finemapping, uppartRect);
		cv::Mat downpartMat(image_finemapping, downPartRect);
		cv::resize(uppartMat, uppartMat, cv::Size(140, 36));
		cv::resize(downpartMat, downpartMat, cv::Size(140, 36));
		std::pair<vector<int>, float> resdown = prc.segmentationFreeRecognizer->SegmentationFreeForSinglePlate(downpartMat);
		std::pair<vector<int>, float> resup = prc.segmentationFreeRecognizer->SegmentationFreeForSinglePlate(uppartMat);
		
		if(resdown.first.size()>5)
			resdown.first.resize(5);

		if(resup.first.size()>2)
		    resup.first.resize(2);
		res.first.insert(res.first.begin(),resup.first.begin(),resup.first.end());
		res.first.insert(res.first.end(),resdown.first.begin(),resdown.first.end());
		res.second = (resdown.second + resup.second) / 2.0;
	}
	if (plateClass == 7)
	{
		cv::Mat image_finemappingWhitePlate;
		cv::resize(image_finemapping, image_finemappingWhitePlate, cv::Size(136, 36));//
		Rect cutPlate;
		cutPlate.x = 2;
		cutPlate.y = 2;
		cutPlate.width = image_finemappingWhitePlate.cols - 4;
		cutPlate.height = image_finemappingWhitePlate.rows - 4;;
		cv::Mat cutPlatemat(image_finemappingWhitePlate, cutPlate);

		whitePlateType = platewhite->classify(cutPlatemat);

		cv::resize(image_finemapping, image_finemapping, cv::Size(140, 60));
		cv::Mat proposal;
		if (image_finemapping.channels() == 3)
		{
			cv::cvtColor(image_finemapping, proposal, cv::COLOR_BGR2GRAY);

			bitwise_not(proposal, proposal);  //yellow white new 
		}
		else
		{
			image_finemapping.copyTo(proposal);

			bitwise_not(proposal, proposal);  //yellow white new 
		}
		//cut the double layer
		int k = 0.01, win = 22;
		Mat imageout = cv::Mat::zeros(image_finemapping.size(), CV_8U);
		util::NiblackSauvolaWolfJolion(proposal, imageout, SAUVOLA, win, win, 0.18 * k);//
		int cutPosition = util::cut2LayerPlatePositionWhite2(imageout);
		if (cutPosition > 35 || cutPosition < 20)cutPosition = 25;

		//cut to two parts
		Rect downPartRect, uppartRect;
		uppartRect.x = 30;
		uppartRect.y = 0;
		uppartRect.width = image_finemapping.cols-60;
		uppartRect.height = cutPosition;//25;

		downPartRect.x = 0;
		downPartRect.y = cutPosition-3;//27;
		downPartRect.width = image_finemapping.cols;
		downPartRect.height = image_finemapping.rows- cutPosition+3;//30;

		cv::Mat uppartMat(image_finemapping, uppartRect);//image_finemapping
		cv::Mat downpartMat(image_finemapping, downPartRect);//image_finemapping
		cv::resize(uppartMat, uppartMat, cv::Size(140, 36));
		cv::resize(downpartMat, downpartMat, cv::Size(140, 36));
		
		bitwise_not(downpartMat, downpartMat);  //yellow white new 
		bitwise_not(uppartMat, uppartMat);  //yellow white new 
		std::pair<vector<int>, float> resdown = prc.segmentationFreeRecognizer->SegmentationFreeForSinglePlate(downpartMat);
		std::pair<vector<int>, float> resup = prc.segmentationFreeRecognizer->SegmentationFreeForSinglePlate(uppartMat);

		if(resdown.first.size()>5)
			resdown.first.resize(5);

		if(resup.first.size()>2)
			resup.first.resize(2);
		res.first.insert(res.first.begin(),resup.first.begin(),resup.first.end());
		res.first.insert(res.first.end(),resdown.first.begin(),resdown.first.end());
		res.second = (resdown.second + resup.second) / 2.0;
	}
#endif

	pair<string,float> result;
	string name = "";
	vector<string> a= pr::CH_PLATE_CODE;
	for(int i=0; i<res.first.size(); ++i)
	{
	    name += pr::CH_PLATE_CODE[res.first[i]];
	}
	result.first = name;
	result.second = res.second;
	return result;
}

PlateInfo Detector::PlateRecogPipeline(unsigned char* imagedata, int width, int height, float confidence_threshold)
{
	PlateInfo results;
	int plateClass;

	clock_t start2, finish2;
	start2 = clock();
	int x = 0, y = 0, w = 0, h = 0;
	cv::Mat plateMat = DetectPlate(imagedata, width, height, x,y,w,h,confidence_threshold, plateClass);
	
	cv::Rect plateRect;
	plateRect.x = x;
	plateRect.y = y;
	plateRect.width = w;
	plateRect.height = h;
	results.setPlateRect(plateRect);

	std::pair<std::string, float> res;
	int whitePlateType=10;
	if (plateMat.data != NULL)
	{
		res = PlateRecogniser(plateMat.data, plateMat.cols, plateMat.rows, plateClass, whitePlateType);
	}

	results.setPlateName(res.first);
	results.confidence = res.second;

	switch (plateClass)
	{
	case 1:
		results.ColorType = BLUE;
		results.IndustryType = SINGALBLUE;
		break;
	case 2:
		results.ColorType = GREEN;
		results.IndustryType = NEWENERGE;
		break;
	case 3:
		results.ColorType = BLACK;
		results.IndustryType = EMBASSYGANGAO;
		break;
	case 4:
		results.ColorType = YELLOW;
		results.IndustryType = DOUBLEYELLOW;
		break;
	case 5:
		results.ColorType = YELLOW;
		results.IndustryType = SINGALYELLOW;
		break;

	case 6:
		results.ColorType = WHITE;
		break;
	case 7:
		results.ColorType = WHITE;
		break;
	case 0:
		results.ColorType = UNKNOWN;
		break;
	default:
		break;
	}
	if (plateClass == 4)//shuanghuang
	{
		string plate = results.name;
		char ch = plate[plate.size()-1];
		if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9'))
		{
			results.IndustryType = DOUBLEYELLOW;
		}
		else
		{
			results.IndustryType = DOUBLEYELLOWGUA;
		}
	}
	if (plateClass == 6)//baidan
	{
		if (whitePlateType == 0)
			results.IndustryType = WHITEPOLICE;
		else if (whitePlateType == 1)
			results.IndustryType = WHITEARMY;
		else if(whitePlateType == 3)
			results.IndustryType = WHITEWUJING;
	}
	if (plateClass == 7)//baishuang
	{
		if (whitePlateType == 2)
			results.IndustryType = DOUBLEARMY;
		else if(whitePlateType == 4)
			results.IndustryType = DOUBLEWUJING;
	}

	return results;
}
Detector::~Detector()
{
	delete prc.plateDetection;
	delete prc.fineMapping;
	delete prc.segmentationFreeRecognizer;
}

IPlateDetectionInterface *CreateDetector(const string& model_path)
{
	string tempstring = model_path;

	if (tempstring[tempstring.size() - 1] == '/')
	{
		tempstring = tempstring.substr(0, tempstring.size() - 1);
	}

    cout<<"tempstring: "<<tempstring<<endl;
    tempstring = tempstring + "/plate";
    cout<<"tempstring: "<<tempstring<<endl;
    
	//detect model path 
	const string& Detectmodel_file = tempstring +"/detector/"+"deploy.prototxt";
	const string& Detectweights_file = tempstring +"/detector/"+"MobileNetSSD_deploy_iter_48000.caffemodel";

	IPlateDetectionInterface *Recognizer1 = new Detector(Detectmodel_file, Detectweights_file, "", "0.5, 0.5, 0.5", 0.007843);
	//white plate 5 type
	const string& Whitemodel_file = tempstring + "/whitePlateD5/" + "whitePlateD5deploy.dat";
	const string& Whiteweights_file = tempstring + "/whitePlateD5/" + "whitePlateD5model.dat";

	platewhite = CreateClassifier(Whitemodel_file, Whiteweights_file);
	//recognsize model path
	prc.initial(
		tempstring + "/recogniser/cascade.xml",
		tempstring + "/recogniser/HorizonalFinemapping.prototxt", tempstring + "/recogniser/HorizonalFinemapping.caffemodel",
		tempstring + "/recogniser/SegmenationFree-Inception.prototxt", tempstring + "/recogniser/SegmenationFree-Inception.caffemodel"
	);

	return Recognizer1;
	
}

void DestroyDetector(IPlateDetectionInterface* Recognizer1)
{
	delete Recognizer1;
	Recognizer1 = NULL;

}
