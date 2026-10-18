import logging

from mo_logs import logger

from cochlea.server import app


class ToMoLogs(logging.Handler):
    """Forward stdlib records (werkzeug requests) into mo-logs."""

    def emit(self, record):
        logger.info("{name}: {message}", name=record.name, message=record.getMessage())


logger.start(logs=[{"log_type": "console"}, {"log_type": "file", "file": "logs/cochlea.log"}])
logging.basicConfig(level=logging.INFO, handlers=[ToMoLogs()])

app.run(port=5000)
