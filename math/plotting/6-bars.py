#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    apples = plt.bar([1, 2, 3], fruit[0], width=0.5, color='red')
    bananas = plt.bar([1, 2, 3], fruit[1], width=0.5, color='yellow',
                      bottom=fruit[0])
    oranges = plt.bar([1, 2, 3], fruit[2], width=0.5, color='#ff8000',
                      bottom=fruit[0]+fruit[1])
    peaches = plt.bar([1, 2, 3], fruit[3], width=0.5, color='#ffe5b4',
                      bottom=fruit[0]+fruit[1]+fruit[2])

    plt.xticks([1, 2, 3], ("Farrah", "Fred", "Felicia"))
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.legend((apples, bananas, oranges, peaches),
               ("apples", "bananas", "oranges", "peaches"))

    plt.show()


if __name__ == '__main__':
    bars()
