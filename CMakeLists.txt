include(FetchContent)

# set minimum cmake version
cmake_minimum_required(VERSION 3.11)

# project name and language
project(example LANGUAGES C CXX)

# require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(boost-cmake)
add_subdirectory(poco)

add_executable(
  opencv-classifiers 
  main.cpp 
  data.cpp
  stacktrace.cpp
  model.cpp
  config.cpp
  static.cpp
  middleware.cpp
  history.cpp

  handler_factory.cpp

  cors_middleware.cpp
  logging_middleware.cpp
  error_middleware.cpp

  statements_handler.cpp
  classify_handler.cpp
  statics_handler.cpp
)

target_link_libraries(
    opencv-classifiers 
    ${OpenCV_LIBS} 

    Boost::system
    Boost::thread
    Boost::log
    Boost::program_options
    Boost::chrono
    Boost::exception

    Poco::Net
    Poco::Util
    Poco::Data
    Poco::DataSQLite
    Poco::Encodings
    Poco::Foundation
    Poco::JSON
    Poco::MongoDB
    Poco::Redis
    Poco::XML
    Poco::Zip
)
