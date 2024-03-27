#include "./config.hpp"
#include "./stacktrace.hpp"
#include "Poco/JSON/Array.h"
#include "Poco/JSON/Object.h"
#include "Poco/JSON/Parser.h"
#include "Poco/SharedPtr.h"
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

config::ConfigReader::ConfigReader(std::string config_path, bool caching)
: m_config_path(std::move(config_path)), m_caching(caching){};

std::string read_file(std::string filename) {
    std::ifstream config_file(filename);
    if (!config_file.good()) {
        std::stringstream ss;
        ss << "File with name '" << filename << "' doesnt exist";
        throw ExceptionWithTrace(ss.str());
    }

    std::stringstream ss;
    ss << config_file.rdbuf();
    return ss.str();
}

config::Server parse_server(Poco::SharedPtr<Poco::JSON::Object> server) {
    return config::Server{
        .address = server->get("address"),
        .timeout = server->get("timeout"),
        .max_queued = server->get("max_queued"),
        .max_threads = server->get("max_threads"),
        .statics_dir = server->get("statics_dir"),
    };
}

config::Database parse_database(Poco::SharedPtr<Poco::JSON::Object> database) {
    return config::Database{
        .connector = database->get("connector"),
        .connection_string = database->get("connection_string"),
    };
}

std::vector<config::Model> parse_models(Poco::SharedPtr<Poco::JSON::Array> arr) {
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

    return models;
}

std::vector<config::Preset> parse_presets(Poco::SharedPtr<Poco::JSON::Array> arr) {
    std::vector<config::Preset> presets;

    for (std::size_t i = 0; i < arr->size(); i++) {
        auto model = arr->getObject(i);

        std::string preset_name = model->get("preset_name");
        std::vector<config::Model> models = parse_models(model->getArray("models"));

        presets.push_back(config::Preset{
        .preset_name = preset_name,
        .models = models,
        });
    }

    return presets;
}

config::Config config::ConfigReader::read_config() {
    if (m_caching && m_cache.has_value()) {
        return m_cache.value();
    }

    Poco::JSON::Parser parser;
    auto obj =
    parser
    .parse(read_file(m_config_path))
    .extract<Poco::JSON::Object::Ptr>();

    m_cache = {
        config::Config{
        .server = parse_server(obj->getObject("server")),
        .database = parse_database(obj->getObject("database")),
        .presets = parse_presets(obj->getArray("presets")),
        }
    };

    return m_cache.value();
};