#include "data.hpp"
#define _GNU_SOURCE = 1

#include "./classify_handler.hpp"
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
#include "config.h"
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

inline char separator() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

int show_image(std::vector<std::string> args) {
    if (args.size() != 2) {
        std::cout << "usage: opencv-classifiers <image_path>" << std::endl;
        return -1;
    }

    cv::Mat image;
    image = cv::imread(args[1], cv::IMREAD_COLOR);
    if (!image.data) {
        std::cout << "No image data" << std::endl;
        return -1;
    }

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    cv::waitKey(0);
    return 0;
}

// ./dnn/example_dnn_classification
// --model=../dnn/models/resnet50.onnx
// --input=../data/squirrel_cls.jpg
// --width=224
// --height=224
// --rgb=true
// --scale="0.003921569"
// --mean="123.675 116.28 103.53"
// --std="0.229 0.224 0.225"
// --crop=true
// --initial_width=256
// --initial_height=256
// --classes=../data/dnn/classification_classes_ILSVRC2012.txt

using namespace Poco::Net;

namespace handlers {

class Ping : public HTTPRequestHandler {
    private:
    void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) override {
        response.add("Content-Type", "application/json");
        std::ostream& body = response.send();
        body << "{"
             << "\"key\""
             << ":"
             << "\"value\""
             << "}";
        body.flush();
        response.setStatus(HTTPServerResponse::HTTP_OK);
    };
};

class Middleware {
    public:
    virtual ~Middleware() = default;
    virtual void before_handle(HTTPServerRequest& request, HTTPServerResponse& response){};
    virtual void after_handle(HTTPServerRequest& request, HTTPServerResponse& response){};
};

class LoggingMiddleware : public Middleware {
    private:
    std::chrono::system_clock::time_point request_start;

    public:
    void before_handle(HTTPServerRequest& request, HTTPServerResponse& response) override {
        request_start = std::chrono::system_clock::now();
    };

    void after_handle(HTTPServerRequest& request, HTTPServerResponse& response) override {
        auto request_end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = request_start - request_end;
        std::time_t end_time = std::chrono::system_clock::to_time_t(request_end);

        const std::size_t buffer_size = 32;
        std::tm* ptm = std::localtime(&end_time);
        std::array<char, buffer_size> buffer{};
        std::strftime(buffer.data(), buffer_size, "%a %b %d %H:%M:%S %Y", ptm);

        std::cout << buffer.data() << " | " << request.getURI() << " | " << response.getStatus() << std::endl;
    }
};

class CorsMiddleware : public Middleware {
    private:
    std::chrono::system_clock::time_point request_start;

    public:
    void before_handle(HTTPServerRequest& request, HTTPServerResponse& response) override {
        response.add("Access-Control-Allow-Origin", "*");
    };
};

class Static : public HTTPRequestHandler {
    private:
    const std::string statics_dir;
    std::unordered_map<std::string, std::string> mime_types;
    std::vector<Middleware*> middleware;

    public:
    Static(const std::string statics_dir, std::vector<Middleware*> middleware = {})
    : statics_dir(statics_dir), middleware(std::move(middleware)) {
        mime_types["html"] = "text/html";
        mime_types["js"] = "text/javascript";
        mime_types["css"] = "text/css";
    };

    void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) override {
        for (auto mw : middleware) {
            mw->before_handle(request, response);
        }

        handle_internal(request, response);

        for (auto mw : middleware) {
            mw->after_handle(request, response);
        }
    };

    private:
    void handle_internal(HTTPServerRequest& request, HTTPServerResponse& response) {
        const Poco::URI uri(request.getURI());
        const std::string filepath = resolve_filepath(uri);

        if (!file_exists(filepath)) {
            std::stringstream path_404;
            path_404 << statics_dir << separator() << "404.html";
            response.sendFile(path_404.str(), "text/html");
            response.setStatus(HTTPServerResponse::HTTP_NOT_FOUND);
            return;
        }

        std::string mime_type = "text/plain";
        std::size_t dot_index = filepath.find(".");
        if (dot_index != std::string::npos) {
            auto file_type = filepath.substr(dot_index + 1);
            auto mapped_mime_type = mime_types.find(file_type);
            if (mapped_mime_type != mime_types.end()) {
                mime_type = mapped_mime_type->second;
            }
        }

        response.sendFile(filepath, mime_type);
        response.setStatus(HTTPServerResponse::HTTP_OK);
    }

    std::string resolve_filepath(Poco::URI uri) {
        std::stringstream filepath_st;
        filepath_st << statics_dir;
        const std::string path = uri.getPath();

        if (path == "" || path == "/") {
            filepath_st << "/index.html";
            return filepath_st.str();
        }

        filepath_st << uri.getPath();
        return filepath_st.str();
    }

    bool file_exists(const std::string filename) {
        std::ifstream f(filename.c_str());
        return f.good();
    }
};

class ErrorHandling : public HTTPRequestHandler {
    private:
    HTTPRequestHandler* handler;

    public:
    ErrorHandling(HTTPRequestHandler* handler)
    : handler(handler){};

    private:
    void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) override {
        try {
            handler->handleRequest(request, response);
        } catch (ExceptionWithTrace& ex) {
            response.setStatus(HTTPServerResponse::HTTPStatus::HTTP_INTERNAL_SERVER_ERROR);
            auto& body = response.send();
            body << ex.what() << std::endl
                 << std::endl
                 << "stacktrace:" << std::endl
                 << ex.stacktrace;
            body.flush();
        } catch (std::exception& ex) {
            response.setStatus(HTTPServerResponse::HTTPStatus::HTTP_INTERNAL_SERVER_ERROR);
            auto& body = response.send();
            body << ex.what();
            body.flush();
        }
    };
};

class Factory : public Poco::Net::HTTPRequestHandlerFactory {
    private:
    Poco::Data::Session& session;

    public:
    Factory(Poco::Data::Session& session)
    : session(session){};

    private:
    Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest& request) override {
        if (request.getMethod() == HTTPRequest::HTTP_POST) {
            if (request.getURI() == "/classify") {
                return new ErrorHandling(new ClassifyHandler());
            }
        }

        if (request.getMethod() != HTTPRequest::HTTP_GET) {
            return nullptr;
        }

        if (request.getURI() == "/ping") {
            return new ErrorHandling(new Ping());
        }

        std::string statements_prefix = "/statements";
        if (request.getURI().substr(0, statements_prefix.size()) == statements_prefix) {
            return new ErrorHandling(new StatementsHandler(session));
        }

        return new ErrorHandling(new Static(
        "/Users/plenkinav/Projects/opencv-classifiers/web",
        { new CorsMiddleware(), new LoggingMiddleware() }
        ));
    };
};

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

} // anonymous namespace

class MyServerApplication : public Poco::Util::ServerApplication {
    private:
    int main(const std::vector<std::string>& args) override {
        Poco::Data::SQLite::Connector::registerConnector();
        Poco::Data::Session session("SQLite", "sqlite.sqlite");
        Data::Image::create_table(session);

        HTTPServerParams::Ptr parameters = new HTTPServerParams();
        parameters->setTimeout(10000);
        parameters->setMaxQueued(100);
        parameters->setMaxThreads(4);

        const ServerSocket socket(Socket("localhost:8080"));

        Poco::Net::HTTPServer server(new Factory(session), socket, parameters);

        server.start();
        waitForTerminationRequest();
        server.stopAll();

        return 0;
    };
};
} // namespace handlers

int main(int argc, char** argv) {
    //   cv::dnn::Net net = cv::dnn::readNet(model, config, framework);
    //   net.setPreferableBackend(backendId);
    //   net.setPreferableTarget(targetId);

    handlers::MyServerApplication app;
    return app.run(argc, argv);
}
