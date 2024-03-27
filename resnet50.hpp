#ifndef RESNET50_HPP
#define RESNET50_HPP

#include <fstream>
#include <opencv2/imgproc.hpp>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>

#include "./base64_utils.hpp"
#include "./utils.hpp"


namespace resnet50 {

using namespace std;
using namespace cv;

struct Result {
    int clazz;
    string label;
    float probability;
};

class Resnet50Classifier {
    private:
    cv::dnn::dnn4_v20231225::Net m_net;
    vector<string> m_labels;
    int m_image_width;
    int m_image_height;
    mat_utils::DumpMatToFileHook* m_dump_img_to_file_hook;

    vector<string> load_labels(string labels_filepath) {
        fstream labels_file(labels_filepath);

        vector<string> lines;
        string line;
        while (getline(labels_file, line)) {
            lines.push_back(line);
        }

        return lines;
    }

    Mat preprocess_image(Mat image, string channel_order) {
        resize(image, image, Size(m_image_width, m_image_height));

        if (channel_order == "bgr") {
            cvtColor(image, image, COLOR_BGR2RGB);
        }

        image.convertTo(image, CV_32F);
        image -= cv::Scalar(103.939, 116.779, 123.68);

        if (m_dump_img_to_file_hook != nullptr) {
            m_dump_img_to_file_hook->dump_to_file(image);
        }

        return image;
    }

    vector<float> classify_internal(Mat input) {
        int sz[] = { 1, input.rows, input.cols, 3 };
        Mat blob(4, sz, CV_32F, input.data);
        m_net.setInput(blob);

        Mat result;
        m_net.forward(result);

        vector<float> probabilities;
        for (int col = 0; col < result.cols; col++) {
            probabilities.push_back(result.at<float>(0, col));
        }

        return probabilities;
    }

    vector<Result> postprocess_result(vector<float> probabilities) {
        vector<Result> results;

        for (int i = 0; i < probabilities.size(); i++) {
            results.push_back(Result{
            .clazz = i,
            .label = m_labels[i],
            .probability = probabilities[i],
            });
        }

        return results;
    }

    public:
    Resnet50Classifier(
    string onnx_filepath,
    string labels_filepath,
    int image_width,
    int image_height,
    mat_utils::DumpMatToFileHook* dump_img_to_file_hook = nullptr
    )
    : m_image_width(image_width),
      m_image_height(image_height),
      m_dump_img_to_file_hook(dump_img_to_file_hook) {

        m_net = cv::dnn::readNet(onnx_filepath);
        m_labels = load_labels(labels_filepath);
    };

    vector<Result> classify(string base_64_image, string channel_order) {
        Mat image = base64::decode_image(base_64_image);
        Mat input = preprocess_image(image, channel_order);
        vector<float> output = classify_internal(input);
        return postprocess_result(output);
    }

    vector<Result> classify(Mat image, string channel_order) {
        Mat input = preprocess_image(image, channel_order);
        vector<float> output = classify_internal(input);
        return postprocess_result(output);
    }
};

}; // namespace resnet50

#endif // RESNET50_HPP