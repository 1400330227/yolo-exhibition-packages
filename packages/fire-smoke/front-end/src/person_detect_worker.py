from threading import Thread


class PersonDetectWorker(Thread):

    def __init__(self):
        super(PersonDetectWorker, self).__init__()

