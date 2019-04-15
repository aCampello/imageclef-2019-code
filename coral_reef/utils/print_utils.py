class Printer:
    def __init__(self, verbose):
        self.verbose = verbose

    def __call__(self, string):
        if self.verbose == 1:
            print(string)