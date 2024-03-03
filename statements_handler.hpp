#ifndef STATEMENTS_HANDLER
#define STATEMENTS_HANDLER

#include "Poco/Data/Session.h"
#include "Poco/Net/HTTPRequestHandler.h"

namespace handlers {

class StatementsHandler : public Poco::Net::HTTPRequestHandler {
    private:
    Poco::Data::Session& session;

    public:
    StatementsHandler(Poco::Data::Session& session)
    : session(session){};

    void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) override;
};

} // namespace handlers

#endif // STATEMENTS_HANDLER