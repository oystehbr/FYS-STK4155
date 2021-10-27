import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad
from project1 import helper
from sklearn.metrics import accuracy_score
import seaborn as sns


def SGD(theta_init, eta, C, n_epochs, M,  X, y, gamma=0, tol=1e-14, lmbda=0):
    """
    Stochastic Gradient Descent
    This function will get a better theta value (from the initial
    guess) for optimizing the given cost-function. Returning the 
    new theta values. 

    #TODO
    :param theta_init (np.ndarray):
        the initial guess of our parameters
    :param eta (float): 
        the learning rate for our gradient_descent method
    :param C (function):
        the cost-function we will optimize
    :param N (int):
        number of maximum iterations
    :param tol (float):
        stop the iteration if the distance is less than this tolerance
    :param gamma (float):
        momentum parameter between 0 and 1, (0 if no momentum)

    :return tuple(np.ndarray, int):
        - better estimate of theta, according to the cost-function
        - number of iterations, before returning
    """

    grad_C = egrad(C, 0)

    n = X.shape[0]
    m = int(n/M)

    # If we want the general gradient decent -> two functions in class tho
    # TODO: scaling the learning rate
    # print('start theta')
    # print(theta_init)
    # print(grad_C(theta_init, X, y, lmbda))
    v = eta*grad_C(theta_init, X, y, lmbda)
    print('---------v------------')
    print(v)
    print('-----------v----------')
    theta_previous = theta_init

    # TODO: updating v, will be wrong if no GD, make class Brolmsen
    N = 10
    for i in range(N):
        grad = grad_C(theta_previous, X, y, lmbda)
        # Momentum based GD
        v = gamma*v + eta*grad
        theta_next = theta_previous - v

        # If the iterations are not getting any better
        if np.sum(np.abs(theta_next - theta_previous)) < tol:
            return theta_next, i

        # Updating the thetas
        theta_previous = theta_next

    return theta_next, N

    j = 0
    for epoch in range(n_epochs):
        for i in range(m):
            # Do something with the end interval of the selected
            k = np.random.randint(m)

            # Finding the k-th batch
            xk_batch = X[k*M:(k+1)*M]
            yk_batch = y[k*M:(k+1)*M]
            # grad = grad_C(theta_previous, xk_batch, yk_batch, lmbda)
            grad = grad_C(theta_previous, xk_batch, yk_batch, lmbda)

            v = gamma*v + eta*grad
            theta_next = theta_previous - v

            # TODO: Scaling the learning rate??
            eta = learning_schedule(epoch*i*m)
            # theta_next = theta_previous - eta*grad

            j += 1
            # Check if we have reached the tolerance
            if np.sum(np.abs(theta_next - theta_previous)) < tol:
                print('local')
                return theta_next, j

            # Updating the thetas
            theta_previous = theta_next

    return theta_next, j


def learning_schedule(t):
    return 5/(t+50)
    # TODO: initial learning rate or what??
    # return initial guess/(t+1)


def cost_OLS(beta, X, y):
    """
    cost-function OLS
    # TODO:

    """

    y_pred = X @ beta

    # TODO: make better, maybe have code
    return np.mean((y_pred - y)**2)


def cost_RIDGE(beta, X, y, lmbda):

    y_pred = X @ beta
    return np.mean((y_pred - y)**2) + lmbda*np.sum(beta**2)


def main_OLS():
    n = 500
    x_values, y_values, z_values = helper.generate_data(n)
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=0.2)

    _, _, beta_OLS = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=1, regression_method='OLS'
    )

    X_train = helper.create_design_matrix(x_train, y_train, 1)
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    z_train_scaled = z_train - np.mean(z_train, axis=0)
    X_T = np.matrix.transpose(X_train_scaled)
    res, num = SGD(
        theta_init=np.array([0.0, 1.2, 2.1]),
        eta=1e-1,
        C=cost_OLS,
        n_epochs=100,
        M=10,
        X=X_train_scaled,
        y=z_train_scaled,
        gamma=0.5,
        lmbda=0.1)

    print('---')
    print(f'num: {num}')
    print(f'RIKTIG: {beta_OLS}')
    print(f'Fake: {res}')


def main_RIDGE():
    sns.set()

    n = 500
    x_values, y_values, z_values = helper.generate_data(n, 0)
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=0.2)

    learning_rates = np.logspace(0, -4, 5)
    lmbda_values = np.logspace(0, -6, 7)

    train_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))

    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            eta = 0.01
            lmbda = 0.001
            _, _, beta_RIDGE = helper.predict_output(
                x_train=x_train, y_train=y_train, z_train=z_train,
                x_test=x_test, y_test=y_test,
                degree=1, regression_method='RIDGE', lmbda=lmbda
            )

            X_train = helper.create_design_matrix(x_train, y_train, 1)
            X_test = helper.create_design_matrix(x_test, y_test, 1)
            X_train_scaled = X_train - np.mean(X_train, axis=0)
            X_test_scaled = X_test - np.mean(X_train, axis=0)

            z_train_scaled = z_train - np.mean(z_train, axis=0)
            theta, num = SGD(
                theta_init=np.array([0.0, -1.0, 1.0]),
                eta=eta,
                C=cost_RIDGE,
                n_epochs=100,
                M=20,
                X=X_train_scaled,
                y=z_train_scaled,
                gamma=0.5,
                lmbda=lmbda)

            print('compare:')
            print(f"real: {beta_RIDGE}")
            print(f"fake: {theta}")

            exit()
            z_pred_train = X_train_scaled @ theta + np.mean(z_train, axis=0)
            z_pred_test = X_test_scaled @ theta + np.mean(z_train, axis=0)

            train_R2_score[i][j] = helper.r2_score(z_train, z_pred_train)
            test_R2_score[i][j] = helper.r2_score(z_test, z_pred_test)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(train_R2_score, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_R2_score, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()


if __name__ == '__main__':
    # main_OLS()
    main_RIDGE()
