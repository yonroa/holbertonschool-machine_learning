#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

apples = plt.bar([1, 2, 3], fruit[0], width=0.5, color='red')
bananas = plt.bar([1, 2, 3], fruit[1], width=0.5,
                  color="yellow", bottom=fruit[0])
oranges = plt.bar([1, 2, 3], fruit[2], width=0.5,
                  color="#ff8000", bottom=fruit[0] + fruit[1])
peaches = plt.bar([1, 2, 3], fruit[3], width=0.5,
                  color="#ffe5b4", bottom=fruit[0] + fruit[1] + fruit[2])
plt.xticks(np.arange(1, 4, 1), ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 80.01, step=10))
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend((apples, bananas, oranges, peaches),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.show()
