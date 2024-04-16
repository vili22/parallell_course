#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

void mf(int ny, int nx, int hy, int hx, const float *in, float *out);

int main() {

    CV_32FC3;
    cv::Mat input = imread("/home/vvirkkal/Development/misc/parallel/data/1a.png", cv::ImreadModes::IMREAD_COLOR);
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

    input.convertTo(input, CV_32FC3);    
    float *data = (float*)input.data;    

    cv::cvtColor(input, input, cv::COLOR_RGB2BGR);
    cv::normalize(input, input, 0,255, cv::NORM_MINMAX, CV_8UC3);
    cv::imshow("output", input);
    cv::waitKey();


    std::unique_ptr<float[]> out = std::make_unique<float[]>(input.rows * input.cols * 3);
    mf(input.rows, input.cols, 3, 3, data, out.get());

    cv::Mat resultImage = cv::Mat(input.rows, input.rows, CV_32FC3, out.get());
    cv::normalize(resultImage, resultImage, 0,255, cv::NORM_MINMAX, CV_8UC3);
    cv::imshow("output", resultImage);
    cv::waitKey();
}