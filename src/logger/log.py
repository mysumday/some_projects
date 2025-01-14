import logging
from os import environ


class LogFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG   : "\033[94m",   # Blue
        logging.INFO    : "\033[92m",    # Green
        logging.WARNING : "\033[93m", # Yellow
        logging.ERROR   : "\033[91m",   # Red
        logging.CRITICAL: "\033[95m" # Magenta
    }

    RESET = "\033[0m"  # Reset color

    def format(self, record):
        # Get the original log message
        log_message = super().format(record)

        # Get the color based on the log level
        color = self.COLORS.get(record.levelno, self.RESET)

        # Return the colored log message
        return f"{color}{log_message}{self.RESET}"


DEBUG: bool = "TRUE" == environ.get("DEBUG", "TRUE")

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
handler = logging.StreamHandler()
formatter = LogFormatter("%(levelname)s - %(message)s - %(module)s - %(lineno)d")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False