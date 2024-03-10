#ifndef MODEL_HPP
#define MODEL_HPP

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>

namespace models {

class ImageSource {
    public:
    virtual ~ImageSource() = default;
    virtual cv::Mat to_mat() = 0;
};

class Base64Image : public ImageSource {
    cv::Mat m_decoded;

    public:
    Base64Image(std::string base64_encoded);
    cv::Mat to_mat() override;
};

struct ClassifResult {
    std::string class_name;
    double probability;
};

class OnnxModel {
    private:
    cv::dnn::dnn4_v20231225::Net m_net;
    std::vector<std::string> m_class_names;
    int m_image_width;
    int m_image_height;
    bool m_need_transpose;
    bool m_need_softmax;

    public:
    OnnxModel(
    std::string model_path,
    std::string labels_path,
    int image_width,
    int image_height,
    bool need_softmax,
    bool need_transpose
    );
    ~OnnxModel() = default;

    std::vector<ClassifResult> classify(ImageSource& img_source);
};

}; // namespace models

#endif // MODEL_HPP