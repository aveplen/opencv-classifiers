#include "./middleware.hpp"

handlers::Middleware::Middleware(
Poco::Net::HTTPRequestHandler* base_handler
)
: m_base_handler(base_handler){};

handlers::Middleware::Middleware(){};

handlers::Middleware::~Middleware() {
    delete m_base_handler;
};
