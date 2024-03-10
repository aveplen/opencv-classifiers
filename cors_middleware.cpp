#include "./cors_middleware.hpp"
#include "middleware.hpp"

handlers::CorsMiddleware::CorsMiddleware(
Poco::Net::HTTPRequestHandler* base_handler
)
: handlers::Middleware(base_handler){};

void handlers::CorsMiddleware::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    response.add("Access-Control-Allow-Origin", "*");
    m_base_handler->handleRequest(request, response);
};