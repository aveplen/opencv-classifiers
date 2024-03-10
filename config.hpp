#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <optional>
#include <string>
#include <vector>

namespace config {

struct Model {
    std::string model_name;
    std::string onnx_path;
    std::string labels_path;
    int img_width;
    int img_height;
    bool need_transpose;
    bool need_softmax;
};

struct Config {
    std::vector<Model> models;
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