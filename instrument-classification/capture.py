import sys
from io import StringIO

# source: http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
# Adapted to return the original string, not a list of lines.

class CaptureStdout():
    """
    Allows to capture stdout to a string (a bit like shell redirect).

    Example usage:

    # a third-party function that only prints to stdout
    def print_to_stdout():
        print('Hello, world!')

    with CaptureStdout() as output:
        print_to_stdout()

    # let's write the output to a file
    with open('hello.txt', 'w') as f
        f.write(str(output))
    """
    def __enter__(self):
        self.orig_stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self.orig_stdout
    def __str__(self):
        return self._stringio.getvalue()
    def __repr__(self):
        return str(self)

def test():
    with CaptureStdout() as output:
        print('Hello,')
        print('world!')
    assert str(output) == 'Hello,\nworld!\n'
