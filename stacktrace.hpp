#include <boost/stacktrace/stacktrace_fwd.hpp>
#include <cstring>
#include <string>
#define _GNU_SOURCE = 1

#include "boost/stacktrace.hpp"
#include <exception>
#include <iostream>
#include <optional>

class ExceptionWithTrace : public std::exception {
    public:
    boost::stacktrace::stacktrace stacktrace;
    const std::string message;

    ExceptionWithTrace(std::exception& base_exception);
    ExceptionWithTrace(std::exception& base_exception, const std::string message);
    const char* what() {
        return message.c_str();
    }

    private:
    void init(std::exception& base_exception);
};
