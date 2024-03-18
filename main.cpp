#include "./classify_handler.hpp"
#include "./handler_factory.hpp"
#include "./history.hpp"
#include "./stacktrace.hpp"
#include "./statements_handler.hpp"
#include "Poco/Data/SQLite/Connector.h"
#include "Poco/Data/Session.h"
#include "Poco/Data/Statement.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/ServerSocketImpl.h"
#include "Poco/Path.h"
#include "Poco/Util/ServerApplication.h"
#include "boost/stacktrace.hpp"
#include "config.hpp"
#include "data.hpp"
#include "history.hpp"
#include <Poco/URI.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


#define _GNU_SOURCE = 1


namespace {

class ServerSocketImpl : public Poco::Net::ServerSocketImpl {
    public:
    using Poco::Net::SocketImpl::init;
};

class Socket : public Poco::Net::Socket {
    public:
    Socket(const std::string& address)
    : Poco::Net::Socket(new ServerSocketImpl()) {
        const Poco::Net::SocketAddress socket_address(address);
        auto* socket = dynamic_cast<ServerSocketImpl*>(impl());
        socket->init(socket_address.af());
        socket->setReuseAddress(true);
        socket->setReusePort(false);
        socket->bind(socket_address, false);
        socket->listen();
    }
};

class MyServerApplication : public Poco::Util::ServerApplication {
    private:
    int main(const std::vector<std::string>& args) override {
        config::ConfigReader config_reader("./config.json", false);
        config::Config config = config_reader.read_config();

        Poco::Data::SQLite::Connector::registerConnector();
        Poco::Data::Session session(config.database.connector, config.database.connection_string);
        Data::Image::create_table(session);
        data::HistoryDB::create_table(session);
        data::HistoryEntryDB::create_table(session);
        data::HistoryEntryResultDB::create_table(session);

        Poco::Net::HTTPServerParams::Ptr parameters = new Poco::Net::HTTPServerParams();
        parameters->setTimeout(config.server.timeout);
        parameters->setMaxQueued(config.server.max_queued);
        parameters->setMaxThreads(config.server.max_threads);

        const Poco::Net::ServerSocket socket(Socket(config.server.address));
        handlers::Factory factory(session, config_reader);
        Poco::Net::HTTPServer server(&factory, socket, parameters);

        server.start();
        waitForTerminationRequest();
        server.stopAll();

        return 0;
    };
};

} // anonymous namespace

int main(int argc, char** argv) {
    MyServerApplication app;
    return app.run(argc, argv);
}
