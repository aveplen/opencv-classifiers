#ifndef CNN_SVM_HPP
#define CNN_SVM_HPP

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


namespace cnn_svm {

using namespace std;
using namespace cv;

struct Result {
    int clazz;
    string label;
};

class CNNSVMClassifier {
    private:
    cv::dnn::dnn4_v20231225::Net m_feature_extractor;
    cv::Ptr<cv::ml::SVM> m_svm;
    vector<string> m_labels;
    int m_image_width;
    int m_image_height;
    mat_utils::DumpMatToFileHook* m_dump_img_to_file_hook;
    mat_utils::DumpMatToFileHook* m_dump_features_to_file_hook;

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

    float classify_internal(Mat input) {
        int sz[] = { 1, input.rows, input.cols, 3 };
        Mat blob(4, sz, CV_32F, input.data);
        m_feature_extractor.setInput(blob);
        Mat features = m_feature_extractor.forward();

        if (m_dump_features_to_file_hook != nullptr) {
            m_dump_features_to_file_hook->dump_to_file(features);
        }

        Mat result;
        m_svm->predict(features, result);
        return result.at<float>(0, 0);
    }

    Result postprocess_result(float output) {
        int clazz = (int)output;
        string label = m_labels[clazz];
        return Result{
            .label = label,
            .clazz = clazz,
        };
    }

    public:
    CNNSVMClassifier(
    string onnx_filepath,
    string dat_filepath,
    string labels_filepath,
    int image_width,
    int image_height,
    mat_utils::DumpMatToFileHook* dump_img_to_file_hook = nullptr,
    mat_utils::DumpMatToFileHook* dump_features_to_file_hook = nullptr
    )
    : m_image_width(image_width),
      m_image_height(image_height),
      m_dump_img_to_file_hook(dump_img_to_file_hook),
      m_dump_features_to_file_hook(dump_features_to_file_hook) {

        m_feature_extractor = cv::dnn::readNet(onnx_filepath);
        m_svm = ml::SVM::load(dat_filepath);
        m_labels = load_labels(labels_filepath);
    };

    Result classify(string base_64_image, string channel_order) {
        Mat image = base64::decode_image(base_64_image);
        Mat input = preprocess_image(image, channel_order);
        float output = classify_internal(input);
        return postprocess_result(output);
    }

    Result classify(Mat image, string channel_order) {
        Mat input = preprocess_image(image, channel_order);
        float output = classify_internal(input);
        return postprocess_result(output);
    }
};

}; // namespace cnn_svm

#endif // CNN_SVM_HPP