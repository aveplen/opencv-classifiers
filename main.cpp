#include <opencv2/opencv.hpp>
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: opencv-classifiers <image_path>\n");
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (!image.data) {
        printf ("No image data \n");
        return -1;
    }

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    cv::waitKey(0);
    return 0;
}
