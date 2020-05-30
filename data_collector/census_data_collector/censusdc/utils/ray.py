"""
Class to trick python into installing into windows without ray support.

These methods are placeholders
"""


def remote(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
    return wrapper


def get(*args):
    return


def put(*args):
    return