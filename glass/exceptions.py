import textwrap

class GLInputError(Exception):
    def __init__(self, msg):
        #msg = textwrap.fill(msg, 80)
        Exception.__init__(self, msg)
