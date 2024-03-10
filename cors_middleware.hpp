#ifndef CORS_MIDDLEWARE_HPP
#define CORS_MIDDLEWARE_HPP

#include "./middleware.hpp"

namespace handlers {

class CorsMiddleware : public handlers::Middleware {
    public:
    CorsMiddleware(Poco::Net::HTTPRequestHandler* base_handler);

    void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) override;
};

} // namespace handlers


#endif // CORS_MIDDLEWARE_HPP