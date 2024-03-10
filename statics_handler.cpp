#include "./statics_handler.hpp"

handlers::StaticsHandler::StaticsHandler(
utils::StaticUtils static_utils
)
: m_static_utils(static_utils){};

void handlers::StaticsHandler::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    const Poco::URI uri(request.getURI());
    std::string filepath = m_static_utils.resolve_filepath(uri);
    m_static_utils.send_file(response, filepath);
};