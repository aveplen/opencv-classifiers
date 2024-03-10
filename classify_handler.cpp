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
#include <utility>
#include <vector>

struct ClassifyRequest {
    std::string data;
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

std::vector<ClassifyResponse>
classify(std::vector<config::Model> models_settings, ClassifyRequest req) {
    std::vector<ClassifyResponse> models_results;

    for (config::Model& model_settings : models_settings) {
        models::OnnxModel model(
        model_settings.onnx_path,
        model_settings.labels_path,
        model_settings.img_width,
        model_settings.img_height,
        model_settings.need_softmax,
        model_settings.need_transpose
        );

        models::Base64Image image(req.data);
        auto begin = std::chrono::steady_clock::now();
        std::vector<models::ClassifResult> results = model.classify(image);
        auto end = std::chrono::steady_clock::now();

        long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        std::vector<ClassifyClassProb> response_probs;
        for (auto res : results) {
            response_probs.push_back(ClassifyClassProb{
            .prob = res.probability,
            .clazz = res.class_name,
            });
        }

        models_results.push_back(
        ClassifyResponse{
        .model = model_settings.model_name,
        .probs = response_probs,
        .time_spent = TimeSpan{
        .span = static_cast<unsigned long>(duration),
        .unit = "microsecond",
        } }
        );
    }

    return models_results;
};

handlers::ClassifyHandler::ClassifyHandler(config::Config config)
: m_config(std::move(config)){};

void handlers::ClassifyHandler::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    Poco::JSON::Parser parser;
    auto obj = parser.parse(request.stream())
               .extract<Poco::JSON::Object::Ptr>();

    auto req = ClassifyRequest{
        .data = obj->get("data").extract<std::string>(),
    };

    std::vector<ClassifyResponse> results = classify(m_config.models, req);
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

struct ClassificationContext {
    std::string model_name;
    std::vector<models::ClassifResult> results;
    unsigned long duration;
    std::string duration_unit;
};
