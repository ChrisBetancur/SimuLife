#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>

enum class LogType {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:

    static Logger& getInstance();

    void init(const std::string& filename);

    void log(LogType level, const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

private:
    Logger() = default;
    ~Logger();

    std::ofstream m_logFile;
    std::mutex m_mutex;
    bool initialized = false;

    // Prevent copying
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string getCurrentTime();
    std::string typeToString(LogType level);

};

#endif // LOGGER_H