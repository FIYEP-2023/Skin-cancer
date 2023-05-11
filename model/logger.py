from enum import Enum

LogTypes = Enum("LogTypes", "INFO", "WARNING", "ERROR", "DEBUG")

class Logger:
    @staticmethod
    def log(message: str, log_type: LogTypes = LogTypes.INFO, file: str = "log.txt", print: bool = True):
        with open(file, "a+") as f:
            f.write(f"[{log_type.name}] {message}\n")
        if print:
            print(f"[{log_type.name}] {message}")