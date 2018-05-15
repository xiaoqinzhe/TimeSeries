class Subject:
    def __init__(self):
        self.x = 1
        self.observer = None

    def setX(self, x):
        old = self.x
        self.x = x
        if self.observer:
            self.observer.xChange(old,x)

    def addObserver(self, observer):
        self.observer = observer

class Observer:
    def __init__(self):
        pass

    def xChange(self, old, new):
        print("x of subject changed from {0} to {1}".format(old, new))

subject = Subject()
subject.addObserver(Observer())
subject.setX(2)
