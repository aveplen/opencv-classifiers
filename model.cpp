#include "./model.hpp"
#include "./stacktrace.hpp"
#include "Poco/Base64Decoder.h"
#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace utils {
class Transformation {
    public:
    virtual ~Transformation() = default;
    virtual cv::Mat apply(cv::Mat src) = 0;
};

class Pipeline {
    std::vector<Transformation*> m_stages;

    public:
    Pipeline(std::initializer_list<Transformation*> trans);
    cv::Mat apply(cv::Mat initial_value);
};

class ScaleToFit : public Transformation {
    private:
    cv::Size m_size;

    public:
    ScaleToFit(cv::Size size);
    cv::Mat apply(cv::Mat src) override;
};

class CropToFit : public Transformation {
    private:
    cv::Size m_size;

    public:
    CropToFit(cv::Size size);
    cv::Mat apply(cv::Mat src) override;
};

class Transpose : public Transformation {
    public:
    cv::Mat apply(cv::Mat src) override;
};

class Classify : public Transformation {
    private:
    cv::dnn::dnn4_v20231225::Net m_net;

    public:
    Classify(cv::dnn::dnn4_v20231225::Net net);
    cv::Mat apply(cv::Mat src) override;
};

class SoftMax : public Transformation {
    public:
    cv::Mat apply(cv::Mat src) override;
};

}; // namespace utils

models::Base64Image::Base64Image(std::string base64_encoded) {
    std::istringstream b64str(base64_encoded);
    std::ostringstream decoded;
    Poco::Base64Decoder decoder(b64str);
    std::copy(
    std::istreambuf_iterator<char>(decoder),
    std::istreambuf_iterator<char>(),
    std::ostreambuf_iterator<char>(decoded)
    );

    std::string decoded_str = decoded.str();
    std::vector<char> bytes(decoded_str.begin(), decoded_str.end());

    m_decoded = cv::imdecode(bytes, cv::IMREAD_COLOR);
};

cv::Mat models::Base64Image::to_mat() {
    return m_decoded;
};

std::vector<std::string> read_lines(std::string filename) {
    std::vector<std::string> lines;
    std::ifstream ifstream(filename);

    std::string line;
    while (getline(ifstream, line)) {
        lines.push_back(line);
    }

    return lines;
};

models::OnnxModel::OnnxModel(
std::string model_path,
std::string labels_path,
int image_width,
int image_height,
bool need_softmax,
bool need_transpose
)
: m_net(cv::dnn::readNet(model_path)),
  m_class_names(read_lines(labels_path)),
  m_image_width(image_width),
  m_image_height(image_height),
  m_need_softmax(need_softmax),
  m_need_transpose(need_transpose){};

std::vector<models::ClassifResult> models::OnnxModel::classify(ImageSource& img_source) {
    utils::Pipeline pipeline = {
        new utils::ScaleToFit(cv::Size(m_image_width, m_image_height)),
        new utils::CropToFit(cv::Size(m_image_width, m_image_height)),
        m_need_transpose ? new utils::Transpose() : nullptr,
        new utils::Classify(m_net),
        m_need_softmax ? new utils::SoftMax() : nullptr,
    };

    cv::Mat pred = pipeline.apply(img_source.to_mat());

    std::vector<double> probs;
    pred.reshape(1, 1).copyTo(probs);

    std::vector<models::ClassifResult> results;
    for (std::size_t i = 0; i < probs.size(); i++) {
        results.push_back(models::ClassifResult{
        .probability = probs[i],
        .class_name = m_class_names[i],
        });
    }

    return results;
};

utils::Pipeline::Pipeline(std::initializer_list<utils::Transformation*> trans) {
    for (auto tran : trans) {
        if (tran != nullptr) {
            m_stages.push_back(tran);
        }
    }
}

cv::Mat utils::Pipeline::apply(cv::Mat initial_value) {
    cv::Mat buf = initial_value;
    for (auto stage : m_stages) {
        buf = stage->apply(buf);
    }
    return buf;
};

utils::ScaleToFit::ScaleToFit(cv::Size size)
: m_size(size){};

cv::Mat utils::ScaleToFit::apply(cv::Mat src) {
    try {
        double x_scale_factor = (double)m_size.width / src.cols;
        double y_scale_factor = (double)m_size.height / src.rows;
        double scale_factor = x_scale_factor > y_scale_factor ? x_scale_factor : y_scale_factor;

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(), scale_factor, scale_factor);
        return resized;
    } catch (std::exception& exc) {
        throw ExceptionWithTrace(exc);
    }
};

utils::CropToFit::CropToFit(cv::Size size)
: m_size(size){};

cv::Mat utils::CropToFit::apply(cv::Mat src) {
    if (src.cols < m_size.width || src.rows < m_size.height) {
        throw ExceptionWithTrace("Image is too small to crop");
    }

    try {
        int height_rem = src.rows - m_size.height;
        int width_rem = src.cols - m_size.width;

        return src(
        cv::Range(height_rem / 2, src.rows - (height_rem / 2 + height_rem % 2)),
        cv::Range(width_rem / 2, src.cols - (width_rem / 2 + width_rem % 2))
        );
    } catch (std::exception& exc) {
        throw ExceptionWithTrace(exc);
    }
};

cv::Mat utils::Transpose::apply(cv::Mat src) {
    cv::Mat transposed;
    cv::transpose(src, transposed);

    std::vector<cv::Mat> channels_bgr;
    cv::split(transposed, channels_bgr);

    std::vector<cv::Mat> channels_rgb = { channels_bgr[2], channels_bgr[0], channels_bgr[1] };
    cv::Mat output;
    cv::merge(channels_bgr, output);

    return output;
};

utils::Classify::Classify(cv::dnn::dnn4_v20231225::Net net)
: m_net(net){};

cv::Mat utils::Classify::apply(cv::Mat src) {
    try {
        double scale = 1.0 / 255;
        cv::Size size = cv::Size(src.cols, src.rows);
        cv::Scalar mean(104, 117, 123); // tbd
        bool swapRB = false;
        bool crop = false;
        int ddepth = CV_32F;
        cv::Mat blob = cv::dnn::blobFromImage(src, scale, size, mean, swapRB, crop, ddepth);

        m_net.setInput(blob);
        return m_net.forward();
    } catch (std::exception& exc) {
        throw ExceptionWithTrace(exc);
    }
};

cv::Mat utils::SoftMax::apply(cv::Mat src) {
    try {
        cv::Mat softmax_output;
        exp(src, softmax_output);
        normalize(softmax_output, softmax_output, 1, 0, cv::NORM_L1);
        return softmax_output;
    } catch (std::exception& exc) {
        throw ExceptionWithTrace(exc);
    }
};