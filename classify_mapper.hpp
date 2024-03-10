#ifndef CLASSIFY_MAPPER_HPP
#define CLASSIFY_MAPPER_HPP

#include "Poco/JSON/Object.h"
#include <string>

struct ClassifyRequest {
    std::string data;
};

// =======================

struct ClassifyClassProb {
    std::string clazz;
    double prob;
    Poco::JSON::Object toJson();
};

struct TimeSpan {
    unsigned long span;
    std::string unit;
    Poco::JSON::Object toJson();
};

struct ClassifyResponse {
    std::string model;
    TimeSpan time_spent;
    std::vector<ClassifyClassProb> probs;
    Poco::JSON::Object toJson();
};

// =======================


#endif // CLASSIFY_MAPPER_HPP
