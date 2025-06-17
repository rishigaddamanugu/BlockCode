class Parent:
    def __init__(self):
        print("Initializing the parent!")

class Child(Parent):
    def __init__(self):
        print("Initializing the child")
        print("TYPE OF CHILD:", type(self))
        print("PARENT OF CHILD:", type(super()))

child = Child()