import logging
import time

logger = logging.getLogger(__name__)


def time_operation(description, operation):
    """
    Function that times the provided operation and returns
    the result of the operation and the duration.
    """
    logger.debug(" Start: %s", description)

    start = time.time()
    result = operation()
    end = time.time()
    duration = end - start

    logger.debug(" End: duration = %f s", duration)

    return result, duration
