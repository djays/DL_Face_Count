from .base import MTCNNCounter

def get_model(name):
    return {
        'MTCNN': MTCNNCounter
    }.get(name, None)