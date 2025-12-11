import logging

def setup_logger(name, log_file="mmsbm.log", level=logging.INFO):
    """Configure or refresh a named logger each time this is called."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Close and remove existing handlers to avoid duplicates and leaks
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger