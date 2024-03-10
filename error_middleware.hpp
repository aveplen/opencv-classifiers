#ifndef ERROR_MIDDLEWARE
#define ERROR_MIDDLEWARE

#include "middleware.hpp"

namespace handlers {

class ErrorMiddleware : public handlers::Middleware {
    public:
    ErrorMiddleware(Poco::Net::HTTPRequestHandler* base_handler);

    void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) override;
};

} // namespace handlers

#endif //  ERROR_MIDDLEWARE
