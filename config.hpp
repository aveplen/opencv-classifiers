#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <optional>
#include <string>
#include <vector>

namespace config {

struct Server {
    std::string address;
    int timeout;
    int max_queued;
    int max_threads;
    std::string statics_dir;
};

struct Database {
    std::string connector;
    std::string connection_string;
};

struct Model {
    std::string model_name;
    std::string onnx_path;
    std::string labels_path;
    int img_width;
    int img_height;
    bool need_transpose;
    bool need_softmax;
};

struct Preset {
    std::string preset_name;
    std::vector<Model> models;
};

struct Config {
    Server server;
    Database database;
    std::vector<Preset> presets;
};

class ConfigReader {
    private:
    std::string m_config_path;
    bool m_caching;
    std::optional<Config> m_cache;

    public:
    ConfigReader(std::string config_path, bool caching);
    Config read_config();
};

}; // namespace config

#endif // CONFIG_HPP