class Perceptron(object):

    def __init__(self):
        self.epochs = 100
        self.learning_rate = 0.01
        self.weight1 = 0
        self.weight2 = 0
        self.bias = 0

        self.truth_table = [
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]

    def activation(self, sum):
        return 1 if sum > 0.0 else 0

    def predict(self, X1, X2):
        sum = ((X1 * self.weight1) + (X2 * self.weight2)) + self.bias
        return self.activation(sum), sum

    def train(self):

        for epoch in range(self.epochs):
            print("-----------------------------------"
                  "-------------------------------------")
            labels_lower_row = ["x1", "x2", "LR", "t",
                                "sum", "act", "dw1", "dw2",
                                "db", "w1", "w2", "b"]
            for label in labels_lower_row:
                print(label.rjust(6, ' '), end="")
            print("\n------------------------------------"
                  "-----------------------------------")

            print("Epoch-" + str(epoch + 1))
            for row in self.truth_table:
                activate, sum = self.predict(row[0], row[1])

                cells = [row[0], row[1], self.learning_rate, row[2],
                         sum, activate]

                change = self.weight1 + self.learning_rate * \
                (row[2] - activate) * row[0]
                cells.append(change)
                self.weight1 = change

                change = self.weight2 + self.learning_rate * \
                (row[2] - activate) * row[1]
                cells.append(change)
                self.weight2 = change

                change = self.bias + self.learning_rate * \
                (row[2] - activate)
                cells.append(change)
                self.bias = change

                cells.append(self.weight1)
                cells.append(self.weight2)
                cells.append(self.bias)

                for cell in cells:
                    print(str(cell).rjust(6, ' '), end="")
                print("\n", end="")

        print("\n\n")

        return True
