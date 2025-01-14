import logging
import sys

# This next bit is grabbed from:
# http://stackoverflow.com/questions/384076/how-can-i-make-the-python-logging-output-to-be-colored  # noqa: E501


def add_coloring_to_emit_ansi(fn):
    # add methods we need to the class
    def new(*args):
        levelno = args[0].levelno
        if levelno >= 50:
            color = "\x1b[31m"  # red
        elif levelno >= 40:
            color = "\x1b[31m"  # red
        elif levelno >= 30:
            color = "\x1b[33m"  # yellow
        elif levelno >= 20:
            color = "\x1b[32m"  # green
        elif levelno >= 10:
            color = "\x1b[35m"  # pink
        else:
            color = "\x1b[0m"  # normal
        ln = color + args[0].levelname + "\x1b[0m"
        args[0].levelname = ln
        return fn(*args)

    return new


level = 40
stream = sys.stderr
ufstring = "%(name)-3s: [%(levelname)-9s] %(asctime)s %(message)s"
cfstring = "%(name)-3s: [%(levelname)-18s] %(asctime)s %(message)s"
logger = logging.getLogger("astrophyics_toolset")


def disable_stream_logging():
    if len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])
    h = logging.NullHandler()
    logger.addHandler(h)


def colorize_logging():
    f = logging.Formatter(cfstring)
    logger.handlers[0].setFormatter(f)
    sh.emit = add_coloring_to_emit_ansi(sh.emit)


def uncolorize_logging():
    try:
        f = logging.Formatter(ufstring)
        logger.handlers[0].setFormatter(f)
        sh.emit = original_emitter
    except NameError:
        # sh and original_emitter are not defined because
        # suppressStreamLogging is True, so we continue since there is nothing
        # to uncolorize
        pass


sh = logging.StreamHandler(stream=stream)
# create formatter and add it to the handlers
formatter = logging.Formatter(ufstring)
sh.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(sh)
logger.setLevel(level)
logger.propagate = False

original_emitter = sh.emit

colorize_logging()

logger.debug("Set log level to %s", level)
