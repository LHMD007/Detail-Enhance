
/*******FFFF111:直方图均衡化*******************/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include"opencv2/opencv.hpp"
using namespace cv;

int main()
{
	Mat srcImage, dstImage;
	srcImage = imread("ROI.jpg", 1);
	if (!srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~！ \n"); return false; }

	// 【2】转为灰度图并显示出来
	cvtColor(srcImage, srcImage, CV_BGR2GRAY);
	namedWindow("原始图", 1);
	imshow("原始图", srcImage);

	// 【3】进行直方图均衡化
	equalizeHist(srcImage, dstImage);

	// 【4】显示结果
	namedWindow("经过直方图均衡化后的图", 1);

	imshow("经过直方图均衡化后的图", dstImage);
	imwrite("equlize.jpg", dstImage);

	// 等待用户按键退出程序
	waitKey(0);
	return 0;

}
/*******FFFF111:直方图均衡化*******************/


/*******FFFF222:空间域图像增强（图像锐化 1 基于拉普拉斯算子）*******************/
////对于求一个锐化后的像素点（sharpened_pixel），这个基于拉普拉斯算子的简单算法主
////要是遍历图像中的像素点，根据领域像素确定其锐化后的值
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
//	//处理边界内部的像素点, 图像最外围的像素点应该额外处理
//	for (int row = 1; row < img.rows - 1; row++)
//	{
//		//前一行像素点
//		const uchar* previous = img.ptr<const uchar>(row - 1);
//		//待处理的当前行
//		const uchar* current = img.ptr<const uchar>(row);
//		//下一行
//
//		const uchar* next = img.ptr<const uchar>(row + 1);
//		uchar *output = result.ptr<uchar>(row);
//		int ch = img.channels();
//		int starts = ch;
//		int ends = (img.cols - 1) * ch;
//		for (int col = starts; col < ends; col++)
//		{
//			//输出图像的遍历指针与当前行的指针同步递增, 以每行的每一个像素点的每一个通道值为一个递增量, 因为要考虑到图像的通道数
//			*output++ = saturate_cast<uchar>(5 * current[col] - current[col - ch] - current[col + ch] - previous[col] - next[col]);
//
//		}
//
//	} //end loop
//
//	  //处理边界, 外围像素点设为 0
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

/*******FFFF222:空间域图像增强（图像锐化 1 基于拉普拉斯算子）*******************/


/*******FFFF333:视网膜用于图像细节增强***有问题,个人暂时没有看懂何处有bug-_-  -__- ~  ****************/
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
/*******FFFF333:视网膜用于图像细节增强*******************/


/*FFFF444拉伸直方图增强对比度*/
//#include<iostream>
//#include<opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
//using namespace std;
//using namespace cv;
//class Histogram1D {
//private:
//	int histSize[1];   //直方图箱子的数量
//	float hranges[2];   //值范围
//	const float *ranges[1]; //值范围的指针
//	int channels[1];  //要检查的通道数量
//public:
//	Histogram1D()
//	{
//		histSize[0] = 256;  //256个箱子
//		hranges[0] = 0.0;  //从0开始（含）
//		hranges[1] = 256.0;  //到256（不含）
//		ranges[0] = hranges;
//		channels[0] = 0;   //先关注通道 0
//	}
//	Mat getHistogram(const Mat &img)
//	{
//		Mat hist;
//		//计算直方图
//		calcHist(&img,
//			1,   //仅为一个图像的直方图
//			channels,  //使用的通道
//			Mat(), //不使用掩码
//			hist, // 作为结果的直方图
//			1,   //这是一维的直方图
//			histSize, //箱子数量
//			ranges   //像素值的范围
//			);
//		return hist;
//	}   //注意这里得到的hist是256行一列的Mat
//
//
//	Mat getHistogramImage(const Mat &img, int zoom = 1)
//	{
//		Mat hist = getHistogram(img);
//		return getImageofHistogram(hist, zoom);
//
//	}
//	static Mat getImageofHistogram(const Mat &hist, int zoom)  //根据直方图数据hist画直方图
//	{
//		double maxVal = 0;
//		double minVal = 0;
//		minMaxLoc(hist, &minVal, &maxVal, 0, 0);
//		cout << "minVal= " << minVal << ", maxVal=" << maxVal << endl;  //这里最小值0  最大值2280  个数！！
//		int histSize = hist.rows;
//		Mat histImg(histSize*zoom, histSize*zoom, CV_8U, Scalar(255));
//		int hpt = static_cast<int>(0.9*histSize);
//		for (int h = 0; h<histSize; h++)
//		{
//			float binVal = hist.at<float>(h);
//			if (binVal>0)
//			{
//				int intensity = static_cast<int>(binVal*hpt / maxVal);   //注意这里的归一化，binVal/maxVal(2280)*hpt  保证在255范围内
//				line(histImg, Point(h*zoom, (histSize)*zoom), Point(h*zoom, (histSize - intensity)*zoom), Scalar(0), zoom);  //注意这里画直线的方法，
//			}                        //前面的point是起始点，后面的终点  背景图是白的，直线用黑的，起始点在最下面，然后减去长度。总感觉这里的histSize（256）需要-1
//		}
//		return histImg;
//	}
//
//
//	static Mat applyLookUp(const Mat &img, Mat &lookup)
//	{
//		Mat result;
//		LUT(img, lookup, result);//LUT是查找表，高效！即用空间换时间  就是需要创建矩阵lookup的查找表规则，把一个像素值映射到另一个像素值
//		return result;
//	}
//
//	Mat stretch(const Mat &img, int minValue = 0) //所谓伸展，就是伸展直方图，使各个像素值平铺均匀，即增加对比度
//	{
//		Mat hist = getHistogram(img);  //得到直方图
//		int imin = 0;
//		for (; imin<histSize[0]; imin++)   //找到最小的横坐标 imin  使得次数大于minValue
//		{
//			if (hist.at<float>(imin)>minValue)
//			{
//				break;
//			}
//		}
//		int imax = histSize[0] - 1;
//		for (; imax >= 0; imax--)    //找到最大的横坐标 imax  使得次数大于minValue
//		{
//			if (hist.at<float>(imax)>minValue)
//			{
//				break;
//			}
//		}                                           //minValue代表的是次数、个数！！！像素值最小（0左右的）以及像素值最大（255左右的）
//													//这些极端的值都比较少，  找到比较少的个数对应的像素值坐标（横坐标）
//		Mat lookup(1, 256, CV_8U);  //LUT查找表的像素重映射的规则
//		for (int i = 0; i<256; i++)      //根据像素值大小划分
//		{
//			if (i<imin)                 //    像素值（横坐标）imin左边的都置为0  //极小的置0
//				lookup.at<uchar>(i) = 0;
//			else if (i>imax)          //像素值（横坐标）右边的都置255  //极大的置255
//				lookup.at<uchar>(i) = 255;
//			else
//				lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin)); //[min,max]重新分配 cvRound为取整  //中间的重新映射
//		}
//		Mat result;
//
//		result = applyLookUp(img, lookup);
//		return result;     //返回处理好的增强的对比度 图片
//
//	}   //这里需要分析下形参传进来的minValue！  如果minValue过大，两边的0，255就会多
//
//		//如果minValue过小，两边的0，255就会少
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
/*FFFF444拉伸直方图增强对比度*/


