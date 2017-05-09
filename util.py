import os
from functools import wraps
from multiprocessing import Process, Queue
import logging, logging.handlers


def in_new_process(func):
    # run function in separate process. useful for scons, where parallelism is thread-based and
    # subject to gil. using this, each thread does it's work in a subprocess, concurrently
    @wraps(func)
    def wrapper(*args, **kwargs):
        def worker(*args, **kwargs):
            # use wrapper to move function return value back via queue, without modifying func
            queue = args[0]
            func = args[1]
            queue.put(func(*args[2:], **kwargs))
        queue = Queue()
        p = Process(target=worker, args=((queue, func) + args), kwargs=kwargs)
        p.start()
        p.join()
        return queue.get()
    return wrapper


def getLogger(target):
    pathdirs = str(target[0]).split(os.path.sep)
    if 'experiments' in pathdirs:
        name = '/'.join(pathdirs[pathdirs.index('experiments'):])
    else:
        name  = pathdirs[-1]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    socketHandler = logging.handlers.SocketHandler('localhost',
            logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    logger.addHandler(socketHandler)

    return logger
