#ifndef CLASSIFY_HPP
#define CLASSIFY_HPP

#include "./config.hpp"
#include "Poco/Data/Session.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"

namespace handlers {

class ClassifyHandler : public Poco::Net::HTTPRequestHandler {
    private:
    config::Config m_config;
    Poco::Data::Session& m_session;

    public:
    ClassifyHandler(
    Poco::Data::Session& session,
    config::Config config
    );

    void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) override;
};

} // namespace handlers

#endif // CLASSIFY_HPP