#ifndef HOG_SVM_HPP
#define HOG_SVM_HPP

#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
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


namespace hog_svm {

using namespace std;
using namespace cv;

struct Result {
    int clazz;
    string label;
};

class HOGSVMClassifier {
    private:
    cv::Ptr<cv::ml::SVM> m_svm;
    HOGDescriptor m_hog;
    vector<string> m_labels;
    int m_image_width;
    int m_image_height;
    mat_utils::DumpMatToFileHook* m_dump_img_to_file_hook;
    mat_utils::DumpMatToFileHook* m_dump_hog_to_file_hook;


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
        resize(image, image, Size(m_image_width, m_image_height), 0, 0, cv::INTER_LINEAR);

        if (channel_order == "bgr") {
            cvtColor(image, image, COLOR_BGR2GRAY);
        } else {
            cvtColor(image, image, COLOR_RGB2GRAY);
        }

        if (m_dump_img_to_file_hook != nullptr) {
            m_dump_img_to_file_hook->dump_to_file(image);
        }

        vector<float> descriptors;
        m_hog.compute(image, descriptors);

        cv::Mat descriptors_mat(1, descriptors.size(), CV_32FC1, descriptors.data());
        if (m_dump_hog_to_file_hook != nullptr) {
            m_dump_hog_to_file_hook->dump_to_file(descriptors_mat);
        }

        return descriptors_mat;
    }

    float classify_internal(Mat input) {
        Mat result;
        m_svm->predict(input, result);
        return result.at<float>(0, 0);
    }

    Result postprocess_result(float output) {
        int clazz = (int)output;
        string label = m_labels[clazz];

        Result res = Result{};
        res.label = label;
        res.clazz = clazz;
        return res;
    };

    public:
    HOGSVMClassifier(
    string dat_filepath,
    string labels_filepath,
    int image_width,
    int image_height,
    mat_utils::DumpMatToFileHook* dump_img_to_file_hook = nullptr,
    mat_utils::DumpMatToFileHook* dump_hog_to_file_hook = nullptr
    )
    : m_image_width(image_width),
      m_image_height(image_height),
      m_dump_img_to_file_hook(dump_img_to_file_hook),
      m_dump_hog_to_file_hook(dump_hog_to_file_hook) {

        m_svm = ml::SVM::load(dat_filepath);
        m_labels = load_labels(labels_filepath);

        m_hog.winSize = Size(image_width, image_height);
        m_hog.cellSize = Size(8, 8);
        m_hog.blockSize = Size(16, 16);
        m_hog.blockStride = Size(8, 8);
        m_hog.nbins = 9;
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
}; // namespace hog_svm

#endif // HOG_SVM_HPP