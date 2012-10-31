import textwrap

class GLInputError(StandardError):
    def __init__(self, msg):
        #msg = textwrap.fill(msg, 80)
        StandardError.__init__(self, msg)
