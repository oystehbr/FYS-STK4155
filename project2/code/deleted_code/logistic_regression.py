# import time
# import helper
# from gradient_descent import SGD
# import autograd.numpy as np


# def cost_logistic_regression(beta, X, y, lmbda=0):
#     """
#     The cost function of the regression method OLS
#     # TODO: can we do the same as in cost_functions.py -> for this method

#     :param beta (np.ndarray):
#         the regression parameters
#     :param X (np.ndarray):
#         input values (dependent variables)
#     :param y (np.ndarray):
#         actual output values
#     :param lmbda (float):
#         do not think about this, it will not be used. Just for simplicity of
#         the code structure of the SGD

#     :return (float):
#         the value of the cost function
#     """
#     # Find the predicted values according to the given betas and input values
#     # TODO: make this functional with
#     n = X.shape[0]
#     total_prob = prob(beta, X[0])**y[0]*(1-prob(beta, X[0]))**(1-y[0])

#     for i in range(1, n):
#         p = prob(beta, X[i])
#         total_prob *= p**y[i]*(1-p)**(1 - y[i])

#     return - np.log(total_prob) + lmbda*np.sum(beta**2)


# def prob(beta, X):
#     """
#     helper function for cost_logistic_regression, will
#     establish the probability:
#         P (y=1 | x, beta)

#     :param beta (np.ndarray):
#         input value
#     :param x (np.ndarray, number):
#         input value

#     :return (np.ndarray, number):
#         function value
#     """
#     # TODO: delete this
#     # np.sum([_beta*x**i for i, _beta in enumerate(beta)])
#     # print(np.sum([_beta*x**i for i, _beta in enumerate(beta)]))
#     # print(beta[0] + beta[1]*x)

#     return np.exp(X @ beta) / (1 + np.exp(X @ beta))


# def main():
#     X_train, X_test, y_train, y_test = helper.load_cancer_data(2)

#     X_train_design = helper.create_design_matrix(
#         X_train[:, 0], X_train[:, 1], degree=1)

#     X_test_design = helper.create_design_matrix(
#         X_test[:, 0], X_test[:, 1], degree=1)

#     theta, num = SGD(
#         X=X_train_design, y=y_train,
#         theta_init=np.array([0.1]*X_train_design.shape[1]),
#         eta=0.01,
#         cost_function=cost_logistic_regression,
#         n_epochs=30, M=10,
#         gamma=0.8,
#         lmbda=1e-4
#     )

#     predicted_values_train = np.where(prob(theta, X_train_design) >= 0.5, 1, 0)
#     predicted_values_test = np.where(prob(theta, X_test_design) >= 0.5, 1, 0)

#     # for pred, y in zip(predicted_values_train, y_train):
#     #     print(f'PRED: {pred}, ACTUAL: {y}, BOOL =======>  {pred == y}')

#     print(
#         f'ACCURACY_train => {helper.accuracy_score(predicted_values_train, y_train)}')
#     print(
#         f'ACCURACY_test => {helper.accuracy_score(predicted_values_test, y_test)}')


# if __name__ == '__main__':
#     a = time.time()
#     main()
#     print('TID: ', end='')
#     print(time.time() - a)
