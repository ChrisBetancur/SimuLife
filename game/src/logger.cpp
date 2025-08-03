#include <logger.h>
#include <iomanip>
#include <ctime>
#include <sstream>

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}


void Logger::init(const std::string& filename) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!initialized) {
        m_logFile.open(filename, std::ios::out | std::ios::app);
        if (!m_logFile.is_open()) {
            throw std::runtime_error("Failed to open log file");
        }
        initialized = true;
        m_logFile << "\n\n----- Log Session Started -----\n";
    }
}

Logger::~Logger() {
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
}

void Logger::log(LogType level, const std::string& message) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!initialized) {
        throw std::runtime_error("Logger not initialized");
    }
    m_logFile << "[" << getCurrentTime() << "] [" << typeToString(level) << "] " << message << "\n";
}

void Logger::debug(const std::string& message) {
    log(LogType::DEBUG, message);
}
void Logger::info(const std::string& message) {
    log(LogType::INFO, message);   
}

void Logger::warning(const std::string& message) {
    log(LogType::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogType::ERROR, message);
}


std::string Logger::getCurrentTime() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Logger::typeToString(LogType level) {
    switch(level) {
        case LogType::DEBUG:   return "DEBUG";
        case LogType::INFO:    return "INFO";
        case LogType::WARNING: return "WARNING";
        case LogType::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}