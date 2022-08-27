import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error


class ArvnImageProcessing:
    @staticmethod
    def get_grayscaled(img):
        pass

class PolynomialRegression(object):
    def __init__(self, degree=1, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


def ransac_polyfit(x_vals, y_vals, poly_degree):
    ransac = RANSACRegressor(PolynomialRegression(degree=poly_degree),
                             residual_threshold=2 * np.std(y_vals),
                             random_state=0)
    ransac.fit(np.expand_dims(x_vals, axis=1), y_vals)
    inlier_mask = ransac.inlier_mask_
    return ransac


def ransac_poly_predict(ransac_model: RANSACRegressor, x):
    y_hat = np.squeeze(ransac_model.predict(np.expand_dims(np.array([x]), axis=1)))
    return y_hat
