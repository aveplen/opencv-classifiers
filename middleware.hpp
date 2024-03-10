#ifndef MIDDLEWARE_HPP
#define MIDDLEWARE_HPP

#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"

namespace handlers {

class Middleware : public Poco::Net::HTTPRequestHandler {
    protected:
    Poco::Net::HTTPRequestHandler* m_base_handler;

    public:
    Middleware(Poco::Net::HTTPRequestHandler* base_handler);
    Middleware();
    ~Middleware();

    virtual void handleRequest(
    Poco::Net::HTTPServerRequest& request,
    Poco::Net::HTTPServerResponse& response
    ) = 0;
};

}; // namespace handlers

#endif // MIDDLEWARE_HPP