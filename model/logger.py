from enum import Enum
from datetime import datetime

LogTypes = Enum("LogTypes", ["INFO", "WARNING", "ERROR", "DEBUG"])

class Logger:
    @staticmethod
    def log(message: str, level = LogTypes.INFO, file: str = "log.txt", should_print: bool = True, end: str = "\n"):
        current_time = datetime.now().strftime("%H:%M:%S")
        with open(file, "a+") as f:
            f.write(f"[{current_time}] [{level.name}] {message}\n")
        if should_print:
            print(f"[{current_time}] [{level.name}] {message}", end=end)