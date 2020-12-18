class MyTest:
    def __init__(self, func):
        self.func = func

    def run(self):
        self.func()


def test_p():
    print("test")


if __name__ == "__main__":
    MyTest(test_p).run()
