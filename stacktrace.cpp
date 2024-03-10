#include "./stacktrace.hpp"
#include <boost/stacktrace/stacktrace_fwd.hpp>
#include <exception>
#include <iostream>

ExceptionWithTrace::ExceptionWithTrace(std::exception& base_exception)
: message(base_exception.what()) {
    init(base_exception);
};

ExceptionWithTrace::ExceptionWithTrace(
std::exception& base_exception,
const std::string message
)
: message(message) {
    init(base_exception);
};

ExceptionWithTrace::ExceptionWithTrace(const std::string message)
: message(message) {
    this->stacktrace = boost::stacktrace::stacktrace();
};

void ExceptionWithTrace::init(std::exception& base_exception) {
    try {
        auto with_trace = dynamic_cast<ExceptionWithTrace&>(base_exception);
        this->stacktrace = with_trace.stacktrace;
        return;
    } catch (std::exception& ex) {
        this->stacktrace = boost::stacktrace::stacktrace();
        return;
    }
};
