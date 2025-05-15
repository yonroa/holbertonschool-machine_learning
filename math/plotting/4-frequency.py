#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram of student scores for a project"""

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=10, edgecolor="black", range=(0, 100))
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.xlim(0, 100)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.ylim(0, 30)
    plt.title("Project A")
    plt.show()


if __name__ == "__main__":
    frequency()
