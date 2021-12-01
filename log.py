# Logging
import logging

# Handler 1
stream = logging.StreamHandler()
stream_format = logging.Formatter("%(message)s")
stream.setFormatter(stream_format)
stream.setLevel(logging.INFO)

# Handler 2
file = logging.FileHandler("log.log")
file.setLevel(logging.INFO)
file_format = logging.Formatter("%(asctime)s:%(message)s", datefmt="%m-%d %H:%M:%S  ")
file.setFormatter(file_format)

# Log
logs = logging.getLogger(__name__)
logs.setLevel(logging.DEBUG)
logs.addHandler(file)
logs.addHandler(stream)

