import os
import logging


class Logger:
    """
        Create logger to save logs during training
        Args:
            logs_dir:
            saved_fn:

        Returns:

        """

    def __init__(self, logs_dir, saved_fn):
        logger_fn = 'logger_{}.txt'.format(saved_fn)
        logger_path = os.path.join(logs_dir, logger_fn)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)
