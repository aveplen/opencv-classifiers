#include "./config.hpp"
#include "./stacktrace.hpp"
#include "Poco/JSON/Object.h"
#include "Poco/JSON/Parser.h"
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

config::ConfigReader::ConfigReader(std::string config_path, bool caching)
: m_config_path(std::move(config_path)), m_caching(caching){};

config::Config config::ConfigReader::read_config() {
    if (m_caching && m_cache.has_value()) {
        return m_cache.value();
    }

    std::ifstream config_file(m_config_path);
    if (!config_file.good()) {
        std::stringstream ss;
        ss << "File with name '" << m_config_path << "' doesnt exist";
        throw ExceptionWithTrace(ss.str());
    }

    std::stringstream ss;
    ss << config_file.rdbuf();
    std::string config_string = ss.str();

    Poco::JSON::Parser parser;
    auto obj = parser.parse(config_string)
               .extract<Poco::JSON::Object::Ptr>();
    std::vector<config::Config> config_nodes;
    auto arr = obj->getArray("models");

    std::vector<config::Model> models;
    for (std::size_t i = 0; i < arr->size(); i++) {
        auto model = arr->getObject(i);

        models.push_back(config::Model{
        .model_name = model->get("model_name"),
        .onnx_path = model->get("onnx_path"),
        .labels_path = model->get("labels_path"),
        .img_width = model->get("img_width"),
        .img_height = model->get("img_height"),
        .need_transpose = model->get("need_transpose"),
        .need_softmax = model->get("need_softmax"),
        });
    }

    return config::Config{
        .models = models,
    };
};