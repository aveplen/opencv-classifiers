#include "./classify_handler.hpp"
#include "./model.hpp"
#include "./stacktrace.hpp"
#include "Poco/Base64Decoder.h"
#include "Poco/Dynamic/Var.h"
#include "Poco/JSON/Array.h"
#include "Poco/JSON/JSON.h"
#include "Poco/JSON/Object.h"
#include "Poco/JSON/Parser.h"
#include "poco/Foundation/src/zconf.h"
#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <vector>

std::vector<char> decodeBase64(std::string encoded) {
    std::istringstream b64str(encoded);
    std::ostringstream decoded;
    Poco::Base64Decoder decoder(b64str);
    std::copy(
    std::istreambuf_iterator<char>(decoder),
    std::istreambuf_iterator<char>(),
    std::ostreambuf_iterator<char>(decoded)
    );

    std::string decoded_str = decoded.str();
    std::vector<char> bytes(decoded_str.begin(), decoded_str.end());
    return bytes;
}

struct ClassifyRequest {
    std::string data;
    int width;
    int height;
    bool use_pipeline;
};

struct ClassifyClassProb {
    std::string clazz;
    double prob;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;
        result.set("class", this->clazz);
        result.set("prob", this->prob);
        return result;
    }
};

struct TimeSpan {
    unsigned long span;
    std::string unit;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;
        result.set("span", this->span);
        result.set("unit", this->unit);
        return result;
    }
};

struct ClassifyResponse {
    std::string model;
    TimeSpan time_spent;
    std::vector<ClassifyClassProb> probs;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;
        result.set("model", this->model);
        result.set("time_spent", this->time_spent.toJson());
        {
            Poco::JSON::Array probs;
            for (auto& prob : this->probs) {
                probs.add(prob.toJson());
            }
            result.set("probs", probs);
        }

        return result;
    }
};

cv::Mat crop_to_center(cv::Mat original, std::size_t width, std::size_t height) {
    if (original.size().width < width) {
        throw ExceptionWithTrace("Image width is less then horizontal crop bound");
    }

    if (original.size().height < height) {
        throw ExceptionWithTrace("Image height is less then vertical crop bound");
    }

    std::size_t horz_pad = (original.size().width - width) / 2;
    std::size_t vert_pad = (original.size().width - width) / 2;

    return original(
    cv::Range((int)horz_pad, (int)original.size().width - horz_pad),
    cv::Range((int)vert_pad, (int)original.size().height - vert_pad)
    );
}

std::vector<ClassifyResponse> classify(ClassifyRequest req);

void handlers::ClassifyHandler::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    Poco::JSON::Parser parser;
    auto obj = parser.parse(request.stream())
               .extract<Poco::JSON::Object::Ptr>();

    auto req = ClassifyRequest{
        .data = obj->get("data").extract<std::string>(),
        .use_pipeline = obj->get("use_pipeline").extract<bool>(),
        // .width = obj->get("width").extract<int>(),
        // .height = obj->get("height").extract<int>(),
    };

    std::vector<ClassifyResponse> results = classify(req);
    Poco::JSON::Array results_json;
    for (auto& result : results) {
        results_json.add(result.toJson());
    }

    response.set("Content-Type", "application/json");
    response.setStatus(Poco::Net::HTTPServerResponse::HTTP_OK);
    auto& body = response.send();
    results_json.stringify(body);
    body.flush();
    std::cout << "classify 200" << std::endl;
};

std::vector<ClassifyResponse> classify(ClassifyRequest req) {
    if (req.use_pipeline) {
        std::string model_path = "/Users/plenkinav/Projects/opencv-classifiers/models/resnet50-caffe2-v1-9.onnx";
        std::string labels_path = "/Users/plenkinav/Projects/opencv-classifiers/models/synset.txt";
        bool need_softmax = false;

        models::OnnxModel model(model_path, labels_path, need_softmax);
        models::Base64Image image(req.data);

        auto begin = std::chrono::steady_clock::now();
        std::vector<models::ClassifResult> results = model.classify(image);
        auto end = std::chrono::steady_clock::now();

        std::vector<ClassifyClassProb> response_probs;
        for (auto res : results) {
            response_probs.push_back(ClassifyClassProb{
            .prob = res.probability,
            .clazz = res.class_name,
            });
        }

        long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        return {
            ClassifyResponse{
            .model = "resnet50-caffe2-v1-9.onnx",
            .probs = response_probs,
            .time_spent = TimeSpan{
            .span = static_cast<unsigned long>(duration),
            .unit = "microsecond",
            },
            }
        };
    }

    std::vector<char> bytes = decodeBase64(req.data);

    cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
    cv::imwrite("original.jpg", img);
    std::cout << "shape: " << img.size().width << " " << img.size().height << " " << img.channels() << std::endl;

    // cv::Mat cropped = crop_to_center(img, 256, 256);
    // cv::imwrite("cropped.jpg", cropped);
    // std::cout << "shape: " << cropped.size().width << " " << cropped.size().height << " " << cropped.channels() << std::endl;

    double scale = 0.01;
    cv::Size size(224, 224);
    cv::Scalar mean(104, 117, 123);

    bool swapRB = true;
    bool crop = true;
    int ddepth = CV_32F;
    cv::Mat blob = cv::dnn::blobFromImage(img, scale, size, mean, swapRB, crop, ddepth);

    std::vector<std::string> class_names;
    std::ifstream ifs("/Users/plenkinav/Projects/opencv-classifiers/models/synset.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_names.push_back(line);
    }

    std::string model = "/Users/plenkinav/Projects/opencv-classifiers/models/resnet50-caffe2-v1-9.onnx";
    auto net = cv::dnn::readNet(model);

    net.setInput(blob);
    cv::Mat outputs = net.forward();

    std::cout << "width: " << outputs.size().width << " height: " << outputs.size().height << std::endl;

    cv::Point classIdPoint;
    double final_prob = 0.0;
    minMaxLoc(outputs.reshape(1, 1), 0, &final_prob, 0, &classIdPoint);
    int label_id = classIdPoint.x;

    std::cout << outputs << std::endl;

    // for (std::size_t i = 0; i < outputs.size().width; i++) {
    //     cv::Point point = outputs.at<cv::Point>(0, i);
    //     std::cout << "i: " << i << " point.x: " << point.y << " class_name: " << class_names[point.x] << std::endl;
    // }

    std::cout << "final_prob: " << final_prob << " class id: " << classIdPoint.x << " class: " << class_names[classIdPoint.x] << std::endl;

    // minMaxLoc(outputs.reshape(1, 1), 0, &final_prob, 0, &classIdPoint);

    // cv::Point classIdPoint;
    // double final_prob;
    // int label_id = classIdPoint.x;
    // // Print predicted class.
    // string out_text = format("%s, %.3f", (class_names[label_id].c_str()), final_prob);

    return { ClassifyResponse{} };
};
