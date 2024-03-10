#ifndef STATICS_HANDLER_HPP
#define STATICS_HANDLER_HPP

#include "Poco/Net/HTTPRequestHandler.h"
#include "middleware.hpp"
#include "static.hpp"

namespace handlers {

class StaticsHandler : public Poco::Net::HTTPRequestHandler {
    private:
    utils::StaticUtils m_static_utils;

    public:
    StaticsHandler(utils::StaticUtils static_utils);

    void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) override;
};

}; // namespace handlers

#endif // STATICS_HANDLER_HPP