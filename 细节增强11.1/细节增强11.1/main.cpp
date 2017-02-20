
/*******FFFF111:ֱ��ͼ���⻯*******************/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include"opencv2/opencv.hpp"
using namespace cv;

int main()
{
	Mat srcImage, dstImage;
	srcImage = imread("ROI.jpg", 1);
	if (!srcImage.data) { printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����~�� \n"); return false; }

	// ��2��תΪ�Ҷ�ͼ����ʾ����
	cvtColor(srcImage, srcImage, CV_BGR2GRAY);
	namedWindow("ԭʼͼ", 1);
	imshow("ԭʼͼ", srcImage);

	// ��3������ֱ��ͼ���⻯
	equalizeHist(srcImage, dstImage);

	// ��4����ʾ���
	namedWindow("����ֱ��ͼ���⻯���ͼ", 1);

	imshow("����ֱ��ͼ���⻯���ͼ", dstImage);
	imwrite("equlize.jpg", dstImage);

	// �ȴ��û������˳�����
	waitKey(0);
	return 0;

}
/*******FFFF111:ֱ��ͼ���⻯*******************/


/*******FFFF222:�ռ���ͼ����ǿ��ͼ���� 1 ����������˹���ӣ�*******************/
////������һ���񻯺�����ص㣨sharpened_pixel�����������������˹���ӵļ��㷨��
////Ҫ�Ǳ���ͼ���е����ص㣬������������ȷ�����񻯺��ֵ
//
//#include"opencv2/opencv.hpp"
//#include"opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//using namespace cv;
//
//namespace ggicci
//
//{
//	void sharpen(const Mat& img, Mat& result);
//}
//
//void ggicci::sharpen(const Mat& img, Mat& result)
//
//{
//	result.create(img.size(), img.type());
//	//����߽��ڲ������ص�, ͼ������Χ�����ص�Ӧ�ö��⴦��
//	for (int row = 1; row < img.rows - 1; row++)
//	{
//		//ǰһ�����ص�
//		const uchar* previous = img.ptr<const uchar>(row - 1);
//		//������ĵ�ǰ��
//		const uchar* current = img.ptr<const uchar>(row);
//		//��һ��
//
//		const uchar* next = img.ptr<const uchar>(row + 1);
//		uchar *output = result.ptr<uchar>(row);
//		int ch = img.channels();
//		int starts = ch;
//		int ends = (img.cols - 1) * ch;
//		for (int col = starts; col < ends; col++)
//		{
//			//���ͼ��ı���ָ���뵱ǰ�е�ָ��ͬ������, ��ÿ�е�ÿһ�����ص��ÿһ��ͨ��ֵΪһ��������, ��ΪҪ���ǵ�ͼ���ͨ����
//			*output++ = saturate_cast<uchar>(5 * current[col] - current[col - ch] - current[col + ch] - previous[col] - next[col]);
//
//		}
//
//	} //end loop
//
//	  //����߽�, ��Χ���ص���Ϊ 0
//	result.row(0).setTo(Scalar::all(0));
//	result.row(result.rows - 1).setTo(Scalar::all(0));
//	result.col(0).setTo(Scalar::all(0));
//	result.col(result.cols - 1).setTo(Scalar::all(0));
//}
//
//int main()
//
//{
//
//	Mat lena = imread("ROI1.jpg");
//
//	Mat sharpenedLena;
//
//	ggicci::sharpen(lena, sharpenedLena);
//
//
//	namedWindow("lena", 0);
//	imshow("lena", lena);
//	namedWindow("sharpened lena", 1);
//	imshow("sharpened lena", sharpenedLena);
//	imshow("yauntu", lena);
//	cvWaitKey();
//
//	return 0;
//
//}

/*******FFFF222:�ռ���ͼ����ǿ��ͼ���� 1 ����������˹���ӣ�*******************/


/*******FFFF333:����Ĥ����ͼ��ϸ����ǿ***������,������ʱû�п����δ���bug-_-  -__- ~  ****************/
//#include <iostream>
//#include <cstring>
//#include "opencv2/opencv.hpp"
//#include"opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//using namespace cv;
//using namespace std;
//
//static void help(std::string errorMessage)
//{
//	std::cout << "Program init error : " << errorMessage << std::endl;
//	std::cout << "\nProgram call procedure : retinaDemo [processing mode] [Optional : media target] [Optional LAST parameter: \"log\" to activate retina log sampling]" << std::endl;
//	std::cout << "\t[processing mode] :" << std::endl;
//	std::cout << "\t -image : for still image processing" << std::endl;
//	std::cout << "\t -video : for video stream processing" << std::endl;
//	std::cout << "\t[Optional : media target] :" << std::endl;
//	std::cout << "\t if processing an image or video file, then, specify the path and filename of the target to process" << std::endl;
//	std::cout << "\t leave empty if processing video stream coming from a connected video device" << std::endl;
//	std::cout << "\t[Optional : activate retina log sampling] : an optional last parameter can be specified for retina spatial log sampling" << std::endl;
//	std::cout << "\t set \"log\" without quotes to activate this sampling, output frame size will be divided by 4" << std::endl;
//	std::cout << "\nExamples:" << std::endl;
//	std::cout << "\t-Image processing : ./retinaDemo -image lena.jpg" << std::endl;
//	std::cout << "\t-Image processing with log sampling : ./retinaDemo -image lena.jpg log" << std::endl;
//	std::cout << "\t-Video processing : ./retinaDemo -video myMovie.mp4" << std::endl;
//	std::cout << "\t-Live video processing : ./retinaDemo -video" << std::endl;
//	std::cout << "\nPlease start again with new parameters" << std::endl;
//	std::cout << "****************************************************" << std::endl;
//	std::cout << " NOTE : this program generates the default retina parameters file 'RetinaDefaultParameters.xml'" << std::endl;
//	std::cout << " => you can use this to fine tune parameters and load them if you save to file 'RetinaSpecificParameters.xml'" << std::endl;
//}
//
//
//int main(int argc, char* argv[]) 
//{
//	bool useLogSampling = false; // "log" // check if user wants retina log sampling processing
//	std::string inputMediaType = "-image"; // argv[1]
//	string imageOrVideoName = "1.jpg"; // argv[2]
//
// // declare the retina input buffer... that will be fed differently in regard of the input media
//	Mat inputFrame, image;
//	VideoCapture videoCapture;
//	// in case a video media is used, its manager is declared here
//
//
//	if (!strcmp(inputMediaType.c_str(), "-image"))
//	{
//		std::cout << "RetinaDemo: processing image " << imageOrVideoName << std::endl;
//		inputFrame = imread(imageOrVideoName, 0); // load image in RGB mode
//
//		//cv::imshow("111", inputFrame);
//
//	}
//	else {
//		if (!strcmp(inputMediaType.c_str(), "-video"))
//		{
//			if (useLogSampling)// attempt to grab images from a video capture device
//			{
//				videoCapture.open(0);
//			}
//			else // attempt to grab images from a video filestream
//			{
//				std::cout << "RetinaDemo: processing video stream " << imageOrVideoName << std::endl;
//				videoCapture.open(imageOrVideoName);
//			}
//			// grab a first frame to check if everything is ok
//			videoCapture >> inputFrame;
//		}
//		else
//		{
//			help("bad command parameter");
//			return -1;
//		}
//	}
//	if (inputFrame.empty())
//	{
//		help("Input media could not be loaded, aborting");
//		return -1;
//	}
//
//
//	try
//	{
//		// create a retina instance with default parameters setup, uncomment the initialisation you wanna test
//		Ptr<Retina> myRetina;
//
//		// if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
//		if (useLogSampling)
//		{
//			myRetina = new cv::Retina(inputFrame.size(), true, cv::RETINA_COLOR_BAYER, true, 2.0, 10.0);
//		}
//		else // -> else allocate "classical" retina :
//			myRetina = new cv::Retina(inputFrame.size());
//		//               myRetina = &inputFrame.clone();
//
//		// save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
//		myRetina->write("RetinaDefaultParameters.xml");
//
//		// load parameters if file exists
//		//        myRetina->setup("RetinaSpecificParameters.xml");
//		myRetina->setup("RetinaDefaultParameters.xml");
//
//		//        // reset all retina buffers (imagine you close your eyes for a long time)
//		myRetina->clearBuffers();
//
//		// declare retina output buffers
//		cv::Mat retinaOutput_parvo;
//		cv::Mat retinaOutput_magno;
//
//		// processing loop with no stop condition
//		for (;;)
//		{
//			// if using video stream, then, grabbing a new frame, else, input remains the same
//			if (videoCapture.isOpened())
//				videoCapture >> inputFrame;
//
//			// run retina filter on the loaded input frame
//			myRetina->run(inputFrame);
//
//			// Retrieve and display retina output
//			myRetina->getParvo(retinaOutput_parvo);
//			myRetina->getMagno(retinaOutput_magno);
//			cv::imshow("retina input", inputFrame);
//			cv::imshow("Retina Parvo", retinaOutput_parvo);
//			cv::imshow("Retina Magno", retinaOutput_magno);
//			cv::waitKey(10);
//		}
//	}
//	catch (cv::Exception e)
//	{
//		std::cerr << "Error using Retina : " << e.what() << std::endl;
//	}
//
//	// Program end message
//	std::cout << "Retina demo end" << std::endl;
//	cout <<"00000000000000" << endl;
//	return 0;
//}
/*******FFFF333:����Ĥ����ͼ��ϸ����ǿ*******************/


/*FFFF444����ֱ��ͼ��ǿ�Աȶ�*/
//#include<iostream>
//#include<opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
//using namespace std;
//using namespace cv;
//class Histogram1D {
//private:
//	int histSize[1];   //ֱ��ͼ���ӵ�����
//	float hranges[2];   //ֵ��Χ
//	const float *ranges[1]; //ֵ��Χ��ָ��
//	int channels[1];  //Ҫ����ͨ������
//public:
//	Histogram1D()
//	{
//		histSize[0] = 256;  //256������
//		hranges[0] = 0.0;  //��0��ʼ������
//		hranges[1] = 256.0;  //��256��������
//		ranges[0] = hranges;
//		channels[0] = 0;   //�ȹ�עͨ�� 0
//	}
//	Mat getHistogram(const Mat &img)
//	{
//		Mat hist;
//		//����ֱ��ͼ
//		calcHist(&img,
//			1,   //��Ϊһ��ͼ���ֱ��ͼ
//			channels,  //ʹ�õ�ͨ��
//			Mat(), //��ʹ������
//			hist, // ��Ϊ�����ֱ��ͼ
//			1,   //����һά��ֱ��ͼ
//			histSize, //��������
//			ranges   //����ֵ�ķ�Χ
//			);
//		return hist;
//	}   //ע������õ���hist��256��һ�е�Mat
//
//
//	Mat getHistogramImage(const Mat &img, int zoom = 1)
//	{
//		Mat hist = getHistogram(img);
//		return getImageofHistogram(hist, zoom);
//
//	}
//	static Mat getImageofHistogram(const Mat &hist, int zoom)  //����ֱ��ͼ����hist��ֱ��ͼ
//	{
//		double maxVal = 0;
//		double minVal = 0;
//		minMaxLoc(hist, &minVal, &maxVal, 0, 0);
//		cout << "minVal= " << minVal << ", maxVal=" << maxVal << endl;  //������Сֵ0  ���ֵ2280  ��������
//		int histSize = hist.rows;
//		Mat histImg(histSize*zoom, histSize*zoom, CV_8U, Scalar(255));
//		int hpt = static_cast<int>(0.9*histSize);
//		for (int h = 0; h<histSize; h++)
//		{
//			float binVal = hist.at<float>(h);
//			if (binVal>0)
//			{
//				int intensity = static_cast<int>(binVal*hpt / maxVal);   //ע������Ĺ�һ����binVal/maxVal(2280)*hpt  ��֤��255��Χ��
//				line(histImg, Point(h*zoom, (histSize)*zoom), Point(h*zoom, (histSize - intensity)*zoom), Scalar(0), zoom);  //ע�����ﻭֱ�ߵķ�����
//			}                        //ǰ���point����ʼ�㣬������յ�  ����ͼ�ǰ׵ģ�ֱ���úڵģ���ʼ���������棬Ȼ���ȥ���ȡ��ܸо������histSize��256����Ҫ-1
//		}
//		return histImg;
//	}
//
//
//	static Mat applyLookUp(const Mat &img, Mat &lookup)
//	{
//		Mat result;
//		LUT(img, lookup, result);//LUT�ǲ��ұ���Ч�����ÿռ任ʱ��  ������Ҫ��������lookup�Ĳ��ұ���򣬰�һ������ֵӳ�䵽��һ������ֵ
//		return result;
//	}
//
//	Mat stretch(const Mat &img, int minValue = 0) //��ν��չ��������չֱ��ͼ��ʹ��������ֵƽ�̾��ȣ������ӶԱȶ�
//	{
//		Mat hist = getHistogram(img);  //�õ�ֱ��ͼ
//		int imin = 0;
//		for (; imin<histSize[0]; imin++)   //�ҵ���С�ĺ����� imin  ʹ�ô�������minValue
//		{
//			if (hist.at<float>(imin)>minValue)
//			{
//				break;
//			}
//		}
//		int imax = histSize[0] - 1;
//		for (; imax >= 0; imax--)    //�ҵ����ĺ����� imax  ʹ�ô�������minValue
//		{
//			if (hist.at<float>(imax)>minValue)
//			{
//				break;
//			}
//		}                                           //minValue������Ǵ�������������������ֵ��С��0���ҵģ��Լ�����ֵ���255���ҵģ�
//													//��Щ���˵�ֵ���Ƚ��٣�  �ҵ��Ƚ��ٵĸ�����Ӧ������ֵ���꣨�����꣩
//		Mat lookup(1, 256, CV_8U);  //LUT���ұ��������ӳ��Ĺ���
//		for (int i = 0; i<256; i++)      //��������ֵ��С����
//		{
//			if (i<imin)                 //    ����ֵ�������꣩imin��ߵĶ���Ϊ0  //��С����0
//				lookup.at<uchar>(i) = 0;
//			else if (i>imax)          //����ֵ�������꣩�ұߵĶ���255  //�������255
//				lookup.at<uchar>(i) = 255;
//			else
//				lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin)); //[min,max]���·��� cvRoundΪȡ��  //�м������ӳ��
//		}
//		Mat result;
//
//		result = applyLookUp(img, lookup);
//		return result;     //���ش���õ���ǿ�ĶԱȶ� ͼƬ
//
//	}   //������Ҫ�������βδ�������minValue��  ���minValue�������ߵ�0��255�ͻ��
//
//		//���minValue��С�����ߵ�0��255�ͻ���
//};
//
//void main()
//{
//	Mat img1 = imread("ROI1.jpg", 0);
//	imshow("yuantu", img1);
//	Histogram1D h;
//	Mat streteched = h.stretch(img1, 200);
//	Histogram1D h_stretech;
//	Histogram1D h_src;
//	imshow("h_stretech.", h_stretech.getHistogramImage(streteched));
//	imshow("h_src", h_src.getHistogramImage(img1));
//	imshow("src", img1);
//	imshow("stretn", streteched);
//	waitKey(0);
//}
/*FFFF444����ֱ��ͼ��ǿ�Աȶ�*/


