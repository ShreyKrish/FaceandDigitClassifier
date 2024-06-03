# util.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import sys
import inspect
import heapq
import random
import signal

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self, item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        "Dequeue the earliest enqueued item still in the queue."
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    "Implements a priority queue data structure."
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class PriorityQueueWithFunction(PriorityQueue):
    "Implements a priority queue with the same push/pop signature of the Queue and the Stack classes."
    def __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction
        super().__init__()

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        super().push(item, self.priorityFunction(item))

def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class Counter(dict):
    "A counter keeps track of counts for a set of keys."
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        "Increments all elements of keys by the same count."
        for key in keys:
            self[key] += count

    def argMax(self):
        "Returns the key with the highest value."
        if len(self) == 0:
            return None
        max_key = max(self, key=self.get)
        return max_key

    def sortedKeys(self):
        "Returns a list of keys sorted by their values. Keys with the highest values will appear first."
        return sorted(self, key=self.get, reverse=True)

    def totalCount(self):
        "Returns the sum of counts for all keys."
        return sum(self.values())

    def normalize(self):
        "Edits the counter such that the total count of all keys sums to 1."
        total = float(self.totalCount())
        if total == 0:
            return
        for key in self:
            self[key] /= total

    def divideAll(self, divisor):
        "Divides all counts by divisor"
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        If 'y' is not a Counter, attempt to return a simple multiplication result
        for each element in the Counter.
        """
        if isinstance(y, Counter):
            # Perform dot product only if 'y' is a Counter
            sum = 0
            x = self
            for key in x:
                if key in y:
                    sum += x[key] * y[key]
            return sum
        else:
            # If 'y' is not a Counter, try to multiply all elements by 'y'
            result = Counter()
            for key in self:
                result[key] = self[key] * y
            return result


    def __add__(self, y):
        """
        Adds two counters or adds a scalar to all elements of the counter
        if 'y' is not a Counter.
        """
        if isinstance(y, Counter):
            result = Counter(self)
            for key, value in y.items():
                result[key] += value
            return result
        elif isinstance(y, (int, float)):  # If y is a number, add it to all elements
            return Counter({k: v + y for k, v in self.items()})
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Counter' and '{}'".format(type(y).__name__))


    def __sub__(self, y):
        "Subtracting a counter from another gives a counter with the union of all keys and counts of the second subtracted from counts of the first."
        result = Counter(self)
        for key, value in y.items():
            result[key] -= value
        return result

def raiseNotDefined():
    print("Method not implemented: %s" % inspect.stack()[1][3])
    sys.exit(1)

def normalize(vectorOrCounter):
    "normalize a vector or counter by dividing each value by the sum of all values"
    if isinstance(vectorOrCounter, Counter):
        total = float(sum(vectorOrCounter.values()))
        if total == 0:
            return vectorOrCounter
        return Counter({k: v / total for k, v in vectorOrCounter.items()})
    else:
        total = float(sum(vectorOrCounter))
        if total == 0:
            return vectorOrCounter
        return [x / total for x in vectorOrCounter]

def sample(distribution, values=None):
    "Sample one element from a distribution assuming it's either a list of probabilities or a counter of probabilities."
    if isinstance(distribution, Counter):
        items = list(distribution.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
    choice = random.random()
    i, total = 0, distribution[0]
    while choice > total:
        i += 1
        total += distribution[i]
    return values[i]

def sampleFromCounter(ctr):
    "Return one sample from a counter assuming it's a counter of probabilities."
    items = list(ctr.items())
    return sample([v for k, v in items], [k for k, v in items])

def getProbability(value, distribution, values):
    "Gives the probability of a value under a discrete distribution defined by (distribution, values)."
    total = 0.0
    for prob, val in zip(distribution, values):
        if val == value:
            total += prob
    return total

def flipCoin(p):
    "Returns true with probability p."
    r = random.random()
    return r < p

def chooseFromDistribution(distribution):
    "Takes either a counter or a list of (prob, key) pairs and samples"
    if isinstance(distribution, (dict, Counter)):
        return sample(distribution)
    r = random.random()
    base = 0.0
    for prob, element in distribution:
        base += prob
        if r <= base:
            return element

def nearestPoint(pos):
    "Finds the nearest grid point to a position (discretizes)."
    (current_row, current_col) = pos
    grid_row = int(current_row + 0.5)
    grid_col = int(current_col + 0.5)
    return (grid_row, grid_col)

def sign(x):
    "Returns 1 or -1 depending on the sign of x."
    if x >= 0:
        return 1
    else:
        return -1

def arrayInvert(array):
    "Inverts a matrix stored as a list of lists."
    result = [[] for _ in array]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result

def matrixAsList(matrix, value=True):
    "Turns a matrix into a list of coordinates matching the specified value."
    rows, cols = len(matrix), len(matrix[0])
    cells = []
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == value:
                cells.append((row, col))
    return cells

def pause():
    "Pauses the output stream awaiting user feedback."
    input("<Press enter/return to continue>")

class TimeoutFunction:
    "A decorator class that wraps a function to timeout after a certain period of time."

    def __init__(self, function, timeout):
        "timeout must be at least 1 second."
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutError("Timeout occurred")

    def __call__(self, *args):
        if 'SIGALRM' not in dir(signal):
            return self.function(*args)
        old_handler = signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.timeout)
        try:
            result = self.function(*args)
        finally:
            signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)
        return result
