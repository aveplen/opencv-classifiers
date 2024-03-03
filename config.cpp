#include "config.h"
#include "yaml-cpp/yaml.h"

config::Config config::read_config(std::string filename) {
  YAML::Node config = YAML::LoadFile("config.yaml");
};