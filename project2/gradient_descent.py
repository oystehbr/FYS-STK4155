import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad
import helper
from sklearn.metrics import accuracy_score
import seaborn as sns

# TODO: delete this


def learning_schedule(t):
    return 5/(t+50)


def SGD(X, y, theta_init, eta, cost_function, n_epochs, M, gamma=0, tol=1e-14, lmbda=0):
    """
    Doing the stochastic gradient descent algorithm for updating 
    the initial values (theta) to give a model with better fit
    according to the given cost-function

    :param X (np.ndarray): 
        the input values
    :param y (np.ndarray): 
        the target values (output values)
    :param theta_init (np.ndarray):
        the initial guess of our parameters
    :param eta (float):
        the learning rate for our gradient descent method
    :param cost_function (function):
        the cost-function we want to optimize against
    :param M (int):
        the size of the mini batches for the stochastic
    :param gamma (number):
        the amount of momentum we will be using in the gradient descent
        descent. If gamma is 0, then we will have no momentum. If gamma is 1, then
        we will use "full" momentum instead.
    :param tol (float):
        stop the iteration if the distance in the previous theta values 
        is less than this tolerance

    :return tuple(np.ndarray, int):
        - better estimate of theta, according to the cost-function
        - number of iterations, before returning
    """

    # Using autograd to calculate the gradient
    grad_C = egrad(cost_function)

    # Finding the number of batches
    n = X.shape[0]
    m = int(n/M)

    # TODO: delete either
    # v = eta*grad_C(theta_init, X, y, lmbda)
    v = 0

    theta_previous = theta_init
    j = 0
    for epoch in range(n_epochs):
        for i in range(m):
            # Do something with the end interval of the selected
            k = np.random.randint(m)

            # Finding the k-th batch
            Xk_batch = X[k*M:(k+1)*M]
            yk_batch = y[k*M:(k+1)*M]

            grad = grad_C(theta_previous, Xk_batch, yk_batch, lmbda)

            # Using the gradients and stochastic to update the theta
            v = gamma*v + eta*grad
            theta_next = theta_previous - v

            # eta = learning_schedule(epoch*i*m)

            j += 1

            # Check if we have reached the tolerance
            if np.sum(np.abs(theta_next - theta_previous)) < tol:
                print('local')
                return theta_next, j

            # Updating the thetas
            theta_previous = theta_next

    return theta_next, j


def cost_OLS(beta, X, y, lmbda=0):
    """
    The cost function of the regression method OLS

    :param beta (np.ndarray):
        the regression parameters
    :param X (np.ndarray):
        input values (dependent variables)
    :param y (np.ndarray):
        actual output values
    :param lmbda (float):
        do not think about this, it will not be used. Just for simplicity of
        the code structure of the SGD

    :return (float):
        the value of the cost function
    """

    # Find the predicted values according to the given betas and input values
    y_pred = X @ beta

    return np.mean((y_pred - y)**2)


def cost_RIDGE(beta, X, y, lmbda):
    """
    The cost function of the regression method RIDGE

    :param beta (np.ndarray):
        the regression parameters
    :param X (np.ndarray):
        input values (dependent variables)
    :param y (np.ndarray):
        actual output values
    :param lmbda (float):
        the hyperparameter for RIDGE regression

    :return (float):
        the value of the cost function
    """

    # Find the predicted values according to the given betas and input values
    y_pred = X @ beta

    return np.mean((y_pred - y)**2) + lmbda*np.sum(beta**2)


def main_OLS(x_values, y_values, z_values, list_no_of_minibatches=[10], n_epochs=200, degree=1, gamma=0):
    """
    Performing an analysis of the results for OLS regression, with 
    respect to the input: number of minibatches and epocs. Will check the results
    for learning rates from 10^0, 10^-1, ..., 10^-6.

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param list_no_of_minibatches (list):
        list of number of minibatches that we want to run through
        # TODO: is this a smart idea (this or M??)
    :param n_epochs (int), default = 200:
        the number of epochs that the SGD will be running through
    :param degree (int), default = 1:
        complexity of the model
    :param gamma (float), default = 0:
        the amount of momentum we will be using in the gradient descent
        descent. If gamma is 0, then we will have no momentum. If gamma is 1, then
        we will use "full" momentum instead.
    """

    # Preparing the data
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=0.2)
    X_train = helper.create_design_matrix(x_train, y_train, degree)
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    z_train_scaled = z_train - np.mean(z_train, axis=0)

    # Analytical betas
    _, _, beta_OLS = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=degree, regression_method='OLS'
    )

    # Looping through different learning rates
    learning_rates = np.logspace(0, -6, 7)
    for no_of_minibatches in list_no_of_minibatches:
        batch_size = int(X_train.shape[0]/no_of_minibatches)
        print(f'START: number_of_minibatches: {no_of_minibatches}')
        print('--------------------------------')
        print(
            f'MSE: {cost_OLS(beta_OLS, X_train_scaled, z_train):.6f} (ANALYTICAL)')
        for eta in learning_rates:
            beta_SGD, num = SGD(
                X=X_train_scaled, y=z_train_scaled,
                theta_init=np.array(
                    [0.0] + [0.1]*(X_train_scaled.shape[1] - 1)),
                eta=eta, cost_function=cost_OLS,
                n_epochs=n_epochs, M=batch_size,
                gamma=0)

            print(
                f'MSE: {cost_OLS(beta_SGD, X_train_scaled, z_train):.6f} (NUMERICAL (eta = {eta:6.0e}))')
        print('--------------------------------')


def main_RIDGE(x_values, y_values, z_values, no_of_minibatches=10, n_epochs=200, degree=1, gamma=0):
    """
    Performing an analysis of the results for RIDGE regression, with 
    respect to the input: number of minibatches and epocs. 

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param no_of_minibatches (list):
        the number of minibatches to the stochastic gradient decent
    :param n_epochs (int), default = 200:
        the number of epochs that the SGD will be running through
    :param degree (int), default = 1:
        complexity of the model
    :param gamma (float), default = 0:
        the amount of momentum we will be using in the gradient descent
        descent. If gamma is 0, then we will have no momentum. If gamma is 1, then
        we will use "full" momentum instead.
    """

    # TODO: ridge cost-function do not work properly (maybe)
    sns.set()

    # Preparing the data
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=0.2)
    X_train = helper.create_design_matrix(x_train, y_train, degree)
    X_test = helper.create_design_matrix(x_test, y_test, degree)
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    X_test_scaled = X_test - np.mean(X_train, axis=0)
    z_train_scaled = z_train - np.mean(z_train, axis=0)

    batch_size = int(X_train.shape[0]/no_of_minibatches)

    learning_rates = np.logspace(0, -4, 5)
    lmbda_values = np.logspace(-1, -6, 6)

    train_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))

    # TODO: do some cleaning inside here
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            _, _, beta_RIDGE = helper.predict_output(
                x_train=x_train, y_train=y_train, z_train=z_train,
                x_test=x_test, y_test=y_test,
                degree=degree, regression_method='RIDGE', lmbda=lmbda
            )

            theta, num = SGD(
                theta_init=np.array([0.0] + list(np.random.randn(2))),
                eta=eta,
                cost_function=cost_RIDGE,
                n_epochs=n_epochs,
                M=batch_size,
                X=X_train_scaled,
                y=z_train_scaled,
                gamma=gamma,
                lmbda=lmbda
            )

            # print('compare:')
            # print(f"real: {beta_RIDGE}")
            # print(cost_RIDGE(beta_RIDGE, X_train_scaled, z_train_scaled, lmbda))
            # print(f"fake: {theta}")
            # print(cost_RIDGE(theta, X_train_scaled, z_train_scaled, lmbda))

            z_pred_train = X_train_scaled @ theta + np.mean(z_train, axis=0)
            z_pred_test = X_test_scaled @ theta + np.mean(z_train, axis=0)

            # print('UNSCALED')
            # print('>>>> RIDGE (exact) first, GRADIENT after')
            # print(np.mean((z_pred_train_RIDGE - z_train)**2) + lmbda*np.sum(theta**2))
            # print(np.mean((z_pred_train - z_train)**2) + lmbda*np.sum(beta_RIDGE**2))
            # exit()

            train_R2_score[i][j] = helper.r2_score(z_train, z_pred_train)
            test_R2_score[i][j] = helper.r2_score(z_test, z_pred_test)

    fig, ax = plt.subplots(figsize=(8, 8))
    heat = sns.heatmap(train_R2_score, annot=True, ax=ax, cmap="viridis",
                       xticklabels=lmbda_values, yticklabels=learning_rates)

    # TODO: remove the save fig
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    # plt.savefig("project2/plots/RIDGE_heatmap_training_2.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(test_R2_score, annot=True, ax=ax, cmap="viridis",
                xticklabels=lmbda_values, yticklabels=learning_rates)
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    # plt.savefig("project2/plots/RIDGE_heatmap_testing_2.png")
    plt.show()


if __name__ == '__main__':
    n = 500
    noise = 0.1
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main_OLS(x_values, y_values, z_values)

    # main_RIDGE()
