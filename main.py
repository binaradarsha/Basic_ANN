from math import e

# AND gate
data = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]

# OR gate
# data = [
#     [0, 0, 0],
#     [0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 1]
# ]

lr = 0.1    # Learning rate

yhs = [0] * len(data)

w1 = 0.1
w2 = 0.1
wb = 0.1


def perceptron():
    global w1, w2, wb

    while mean_squard_error() != 0:
        for i, r in enumerate(data):
            x1 = r[0]
            x2 = r[1]
            y = r[2]
            # print(f"{x1}, {x2}, {y}")
            yh = predict(x1, x2)
            yhs[i] = yh
            if y != yh:
                w1 = w1 + lr * (y - yh) * x1
                w2 = w2 + lr * (y - yh) * x2
                wb = wb + lr * (y - yh) * 1

    print(f"w1 = {w1}")
    print(f"w2 = {w2}")
    print(f"wb = {wb}")


def predict(x1, x2):
    sum = x1 * w1 + x2 * w2 + wb
    sigmoid = 1 / (1 + e ** -sum)
    output = 0 if (sigmoid < 0.5) else 1
    return output


def mean_squard_error():
    sum = 0
    for i, r in enumerate(data):
        sum = sum + (r[2] - yhs[i]) ** 2

    return sum / len(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    perceptron()

    for r in data:
        exp = r[2]
        act = predict(r[0], r[1])
        print(f"Expected = {exp}, Actual = {act}, Result = {'Positive' if exp == act else 'Negative'}")
