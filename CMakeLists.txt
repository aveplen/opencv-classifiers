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

# install yaml-cpp library
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
)
FetchContent_GetProperties(yaml-cpp)
if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

# install loguru library
FetchContent_Declare(LoguruGitRepo
	GIT_REPOSITORY "https://github.com/emilk/loguru" # can be a filesystem path
	GIT_TAG        "master"
)
# set any loguru compile-time flags before calling MakeAvailable()
set(LOGURU_WITH_STREAMS TRUE)
FetchContent_MakeAvailable(LoguruGitRepo)

add_subdirectory(boost-cmake)
add_subdirectory(poco)

add_executable(
  opencv-classifiers 
  main.cpp 
  data.cpp
  statements_handler.cpp
  classify_handler.cpp
  stacktrace.cpp
  model.cpp
  config.cpp
)

target_link_libraries(
    opencv-classifiers 
    ${OpenCV_LIBS} 
    yaml-cpp::yaml-cpp 
    loguru::loguru

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

add_executable(test test.cpp)
target_link_libraries(
    test 
    ${OpenCV_LIBS} 
    
    Boost::system
    Boost::thread
    Boost::log
    Boost::program_options
    Boost::chrono

    Poco::Data
    Poco::DataSQLite
    Poco::Encodings
    Poco::Foundation
    Poco::JSON
    Poco::MongoDB
    Poco::Net
    Poco::Redis
    Poco::Util
    Poco::XML
    Poco::Zip
)
