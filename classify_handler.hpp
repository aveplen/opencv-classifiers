#ifndef CLASSIFY_HPP
#define CLASSIFY_HPP

#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"

namespace handlers {

class ClassifyHandler : public Poco::Net::HTTPRequestHandler {
    void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) override;
};

} // namespace handlers

#endif // CLASSIFY_HPP