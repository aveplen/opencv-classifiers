#ifndef LOGGING_MIDDLEWARE_HPP
#define LOGGING_MIDDLEWARE_HPP

#include "./middleware.hpp"
#include "Poco/Net/HTTPRequestHandler.h"

namespace handlers {

class LoggingMiddleware : public handlers::Middleware {
    public:
    LoggingMiddleware(Poco::Net::HTTPRequestHandler* base_handler);

    public:
    void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) override;
};

} // namespace handlers

#endif // LOGGING_MIDDLEWARE_HPP