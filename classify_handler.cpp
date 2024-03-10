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
#include <valarray>
#include <vector>

// =================== request ===================

struct ClassifyRequest {
    std::string data;
    std::optional<int> limit;
};

ClassifyRequest parse_request(std::istream& req_stream) {
    Poco::JSON::Parser parser;
    auto obj = parser.parse(req_stream)
               .extract<Poco::JSON::Object::Ptr>();

    auto req = ClassifyRequest{
        .data = obj->get("data"),
        .limit = std::nullopt,
    };

    if (obj->has("limit")) {
        int limit_value = obj->get("limit");
        req.limit = { limit_value };
    }

    return req;
}

// =================== response ===================

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
    std::string top_full_label;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;
        result.set("model", this->model);
        result.set("time_spent", this->time_spent.toJson());
        result.set("top_full_label", this->top_full_label);
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

namespace {
using namespace std;

vector<string> split_str(string src, string del) {
    vector<string> chunks;

    size_t pos = src.find(del);
    size_t initial_pos = 0;

    while (pos != string::npos) {
        chunks.push_back(src.substr(initial_pos, pos - initial_pos));
        initial_pos = pos + 1;
        pos = src.find(del, initial_pos);
    }

    chunks.push_back(src.substr(initial_pos, min(pos, src.size()) - initial_pos + 1));
    return chunks;
}

string join_strings(vector<string> strings, string del = "") {
    stringstream ss;
    for (size_t i = 0; i < strings.size(); i++) {
        ss << strings[i];
        if (i < strings.size() - 1) {
            ss << del;
        }
    }
    return ss.str();
}

string prepare_label(string label) {
    vector<string> chunks_by_space = split_str(label, " ");
    vector<string> rest(chunks_by_space.begin() + 1, chunks_by_space.end());
    vector<string> chunks_by_comma = split_str(join_strings(rest, " "), ", ");
    return chunks_by_comma[0];
}
} // namespace

struct MoreThenKey {
    inline bool operator()(
    const models::ClassifResult& res1,
    const models::ClassifResult& res2
    ) {
        return res1.probability > res2.probability;
    }
};

ClassifyResponse build_response(
std::string model_name,
unsigned long duration,
std::string unit,
std::vector<models::ClassifResult> results,
std::optional<int> req_limit
) {
    std::vector<ClassifyResponse> models_results;

    std::sort(results.begin(), results.end(), MoreThenKey());
    int limit = req_limit.has_value() ? req_limit.value() : results.size();
    std::vector<models::ClassifResult> slice(results.begin(), results.begin() + limit);

    double prob_sum = 0.0;
    std::vector<ClassifyClassProb> response_probs;
    for (std::size_t i = 0; i < slice.size(); i++) {
        models::ClassifResult res = slice[i];

        prob_sum += res.probability;
        auto class_prob = ClassifyClassProb{
            .prob = res.probability,
            .clazz = prepare_label(res.class_name),
        };

        if (i == slice.size() - 1) {
            prob_sum -= res.probability;
            class_prob.prob = 1.0 - prob_sum;
        }

        response_probs.push_back(class_prob);
    }

    return ClassifyResponse{
        .model = model_name,
        .probs = response_probs,
        .top_full_label = slice[0].class_name,
        .time_spent = TimeSpan{
        .span = static_cast<unsigned long>(duration),
        .unit = unit }
    };
}

// =================== logic ===================

std::vector<ClassifyResponse>
classify(std::vector<config::Model> models_settings, ClassifyRequest req) {
    std::vector<ClassifyResponse> repsonse;

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
        std::vector<models::ClassifResult> cls_results = model.classify(image);
        auto end = std::chrono::steady_clock::now();

        unsigned long dur = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        repsonse.push_back(build_response(
        model_settings.model_name,
        dur,
        "microseconds",
        cls_results,
        req.limit
        ));
    }

    return repsonse;
};

// =================== handler ===================

handlers::ClassifyHandler::ClassifyHandler(config::Config config)
: m_config(std::move(config)){};

void handlers::ClassifyHandler::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    ClassifyRequest req = parse_request(request.stream());

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
