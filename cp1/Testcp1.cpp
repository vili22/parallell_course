#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

void correlate(int ny, int nx, const float *data, float *result);

int main() {


    CV_32FC3;
    cv::Mat input =
      imread("/home/vvirkkal/Development/misc/parallel/data/2d.png",
             cv::ImreadModes::IMREAD_GRAYSCALE);
    //cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

    input.convertTo(input, CV_32FC1);    
    float *data = (float*)input.data;    

    //cv::normalize(input, input, 0,255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imshow("output", input);
    //cv::waitKey();

    std::unique_ptr<float[]> result = std::make_unique<float[]>(input.rows * input.rows);
    correlate(input.rows, input.cols, data, result.get());

    cv::Mat resultImage = cv::Mat(input.rows, input.rows, CV_32FC1, result.get());
    cv::normalize(resultImage, resultImage, 0,255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("output", resultImage);
    cv::waitKey();

  

}