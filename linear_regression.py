import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

class LinearRegression:
    def __init__(self, data=None):
        """
        Data should be an array of [x, y] coordinate pairs.
        """
        self.data = data

    def sample_data(self, samples, max_cor):
        """
        Generates sample data.
        samples = number of data points
        max_cor = maximum x or y coordinate
        """
        r = lambda: np.random.randint(1, max_cor)
        self.data = [[r(), r()] for _ in range(samples)]

    def fit(self):

        # arrays of x and y coordinates
        self.x_cor = [n[0] for n in self.data]
        self.y_cor = [n[1] for n in self.data]
        
        # mean of x and y coordinates
        x_mean = np.mean(self.x_cor)
        y_mean = np.mean(self.y_cor)
        
        # least squares
        dividend = sum([x - x_mean for x in self.x_cor]) * sum([y - y_mean for y in self.y_cor])
        divisor = sum([x - x_mean for x in self.x_cor]) ** 2
        
        # slope of best fit line
        self.m = np.array(dividend/divisor)
        
        # y-intercept
        self.b = np.array(y_mean - (self.m * x_mean))

    def show(self):
        """
        Displays linear regression.
        """
        plt.scatter(self.x_cor, self.y_cor)
        plt.plot(self.x_cor, self.x_cor * self.m + self.b, 'r')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.show()

    def predict(self, point):
        return point[0] * self.m + self.b


if __name__ == '__main__':
    lr = LinearRegression()
    lr.sample_data(samples=100, max_cor=100)
    lr.fit()

    input = [2, 2]
    y = lr.predict(input)
    print('Prediction for {0}: {1}'.format(input, y))

    lr.show()
