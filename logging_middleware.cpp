#include "./logging_middleware.hpp"
#include "middleware.hpp"

handlers::LoggingMiddleware::LoggingMiddleware(
Poco::Net::HTTPRequestHandler* base_handler
)
: handlers::Middleware(base_handler){};

void handlers::LoggingMiddleware::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    auto request_start = std::chrono::system_clock::now();

    m_base_handler->handleRequest(request, response);

    auto request_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = request_start - request_end;
    std::time_t end_time = std::chrono::system_clock::to_time_t(request_end);

    const std::size_t buffer_size = 32;
    std::tm* ptm = std::localtime(&end_time);
    std::array<char, buffer_size> buffer{};
    std::strftime(buffer.data(), buffer_size, "%a %b %d %H:%M:%S %Y", ptm);

    std::cout << buffer.data() << " | " << request.getURI() << " | " << response.getStatus() << std::endl;
};