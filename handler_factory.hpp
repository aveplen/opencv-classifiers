#ifndef HANDLER_FACTORY
#define HANDLER_FACTORY

#include "./config.hpp"
#include "Poco/Data/Session.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"


namespace handlers {

class Factory : public Poco::Net::HTTPRequestHandlerFactory {
    private:
    Poco::Data::Session& m_session;
    config::ConfigReader& m_config_reader;

    public:
    Factory(Poco::Data::Session& session, config::ConfigReader& config_reader);

    Poco::Net::HTTPRequestHandler* createRequestHandler(
    const Poco::Net::HTTPServerRequest& request
    ) override;
};

}; // namespace handlers

#endif // HANDLER_FACTORY