#ifndef DATA_HPP
#define DATA_HPP

#include "Poco/Data/Session.h"
#include "Poco/DateTime.h"
#include "Poco/JSON/Object.h"
#include <cstddef>
#include <string>
#include <vector>

namespace Data {

const std::string IMAGE_CREATE_TABLE_STATEMENT = "CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, width INTEGER NOT NULL, height INTEGER NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL);";
const std::string IMAGE_SELECT_BY_ID_STATEMENT = "SELECT id, height, width, data, created_at FROM images WHERE id = ?";
const std::string IMAGE_SELECT_ALL_STATEMENT = "SELECT id, height, width, data, created_at FROM images;";
const std::string IMAGE_INSERT_STATEMENT = "INSERT INTO images (height, width, data, created_at) VALUES(?, ?, ?, ?) RETURNING id";
const std::string IMAGE_UPDATE_STATEMENT = "UPDATE images SET height = ?, width = ?, data = ? WHERE id = ?;";

struct Image {
    unsigned long int id = 0;
    unsigned int width;
    unsigned int height;
    std::string data;
    Poco::DateTime created_at;

    static void create_table(
    Poco::Data::Session& session
    );

    static std::optional<Image> find(
    Poco::Data::Session& session,
    unsigned long int id
    );

    static std::vector<Image> all(
    Poco::Data::Session& session,
    unsigned int limit = 0
    );

    void update(Poco::Data::Session& session);

    void create(Poco::Data::Session& session);

    Poco::JSON::Object toJson();
};

} // namespace Data

#endif // DATA_HPP