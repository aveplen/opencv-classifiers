#include "./handler_factory.hpp"
#include "./classify_handler.hpp"
#include "./error_middleware.hpp"
#include "./history_handler.hpp"
#include "./logging_middleware.hpp"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "cors_middleware.hpp"
#include "history_handler.hpp"
#include "logging_middleware.hpp"
#include "middleware.hpp"
#include "statements_handler.hpp"
#include "static.hpp"
#include "statics_handler.hpp"

handlers::Factory::Factory(Poco::Data::Session& session, config::ConfigReader& config_reader)
: m_session(session),
  m_config_reader(config_reader){};

Poco::Net::HTTPRequestHandler* handlers::Factory::createRequestHandler(
const Poco::Net::HTTPServerRequest& request
) {
    if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_POST) {
        if (request.getURI() == "/classify") {
            return new ErrorMiddleware(
            new LoggingMiddleware(
            new CorsMiddleware(
            new ClassifyHandler(
            m_session,
            m_config_reader.read_config()
            )
            )
            )
            );
        }
    }

    if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_GET) {
        if (request.getURI() == "/history") {
            return new ErrorMiddleware(
            new LoggingMiddleware(
            new CorsMiddleware(
            new HistoryHandler(
            m_session,
            m_config_reader.read_config()
            )
            )
            )
            );
        }
    }

    if (request.getMethod() != Poco::Net::HTTPRequest::HTTP_GET) {
        return nullptr;
    }

    std::string statements_prefix = "/statements";
    if (request.getURI().substr(0, statements_prefix.size()) == statements_prefix) {
        return new ErrorMiddleware(new StatementsHandler(m_session));
    }

    config::Config config = m_config_reader.read_config();
    utils::StaticUtils static_utils(config.server.statics_dir);

    return new ErrorMiddleware(
    new CorsMiddleware(
    new StaticsHandler(static_utils)
    )
    );
};