"""Base testcase to enforce inheritance chain"""

import logging
import sys
import unittest

logger = logging.getLogger()


def configure_logging():
    hdlr_info = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s][PID:%(process)d, TID:%(thread)d][%(levelname)s]["
        "%(module)s.%(funcName)s:%(lineno)d] %(message)s"
    )
    hdlr_info.setFormatter(formatter)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    logger.addHandler(hdlr_info)
    logger.setLevel(logging.INFO)


configure_logging()


class BaseTestCase(unittest.TestCase):
    """Test base class."""

    def setUp(self):
        super().setUp()
