#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace config {
    struct Config {
        std::string model_name;
        std::string model_path;
        bool preprocess_with_mean;
        bool preprocess_with_scale;
        int preprocess_by_muliplying_width;
        int preprocess_by_muliplying_height;
        bool works_with_rgb;
    };

    Config read_config(std::string filename);
};

#endif // CONFIG_H