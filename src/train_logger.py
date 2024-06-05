import json
import logging
import os


class TrainLogger:

    logger: logging.Logger = None

    def __init__(self, logging_path: str):
        """
        Initialize the TrainLogger with the given logging path.

        @param logging_path: Path where the log file will be saved.
        """
        self.logging_path = logging_path
        # self._setup_logging()

    def _setup_logging(self):
        """
        Set up the logging configuration. This includes setting the logging level,
        format, and adding a file handler to log messages to a file.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        file_handler = self._create_file_handler('train.log')
        self.logger.addHandler(file_handler)

    def _create_file_handler(self, filename):
        """
        Create a file handler for logging messages to a file.

        @param filename: Name of the log file.
        @return: Configured file handler.
        """
        file_handler = logging.FileHandler(os.path.join(self.logging_path, filename))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        return file_handler

    def log_event(self, event: str, level: str = "info", **kwargs):
        """
        Log an event with the specified level and additional keyword arguments.

        @param event: Description of the event to log.
        @param level: Logging level (info, error, warning, debug). Defaults to "info".
        @param kwargs: Additional keyword arguments to include in the log message.
        """
        log_data = {"event": event}
        log_data.update(kwargs)
        log_message = json.dumps(log_data)

        if self.logger is not None:
            if level == "info":
                self.logger.info(log_message)
            elif level == "error":
                self.logger.error(log_message)
            elif level == "warning":
                self.logger.warning(log_message)
            elif level == "debug":
                self.logger.debug(log_message)
            else:
                self.logger.log(logging.NOTSET, log_message)
        else:
            print(log_message)
