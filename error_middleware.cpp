#include "./error_middleware.hpp"
#include "./stacktrace.hpp"
#include "middleware.hpp"

handlers::ErrorMiddleware::ErrorMiddleware(
Poco::Net::HTTPRequestHandler* base_handler
)
: handlers::Middleware(base_handler){};

void handlers::ErrorMiddleware::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    try {
        m_base_handler->handleRequest(request, response);
    } catch (ExceptionWithTrace& ex) {
        response.setStatus(Poco::Net::HTTPServerResponse::HTTPStatus::HTTP_INTERNAL_SERVER_ERROR);
        auto& body = response.send();
        body << ex.what() << std::endl
             << std::endl
             << "stacktrace:" << std::endl
             << ex.stacktrace;
        body.flush();
    } catch (std::exception& ex) {
        response.setStatus(Poco::Net::HTTPServerResponse::HTTPStatus::HTTP_INTERNAL_SERVER_ERROR);
        auto& body = response.send();
        body << ex.what();
        body.flush();
    }
};
