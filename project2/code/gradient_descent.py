from cost_functions import cost_OLS, cost_RIDGE
from autograd import elementwise_grad as egrad
import autograd.numpy as np
import matplotlib.pyplot as plt
import helper
import seaborn as sns


def learning_schedule(t):
    return 5/(t+50)


def SGD(X, y, theta_init, eta, cost_function, n_epochs, batch_size, gamma=0, tol=1e-14, lmbda=0, scale_learning = False):
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
    m = int(n/batch_size)

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
            Xk_batch = X[k*batch_size:(k+1)*batch_size]
            yk_batch = y[k*batch_size:(k+1)*batch_size]

            grad = grad_C(theta_previous, Xk_batch, yk_batch, lmbda)

            # Using the gradients and stochastic to update the theta
            v = gamma*v + eta*grad
            theta_next = theta_previous - v

            if scale_learning:
                eta = learning_schedule(epoch*i*m)

            # Check if we have reached the tolerance
            if np.sum(np.abs(theta_next - theta_previous)) < tol:
                print('local')
                return theta_next, j

            # Updating the thetas
            theta_previous = theta_next

    return theta_previous, j

def main_OLS1(x_values, y_values, z_values, list_no_of_minibatches=[10], n_epochs=200, degree=1, gamma=0):
    #TODO:change despription
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
    X_test = helper.create_design_matrix(x_test, y_test, degree)
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    X_test_scaled = X_test - np.mean(X_train, axis=0)
    z_train_scale = np.mean(z_train, axis=0)
    z_train_scaled = z_train - z_train_scale

    # Analytical betas
    _, _, beta_OLS = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=degree, regression_method='OLS'
    )

    #Testing scaling algoritm
    print("Scaling learning rate alogrithm")
    for i in range(5):
        beta_SGD, num = SGD(
                    X=X_train_scaled, y=z_train_scaled,
                    theta_init=np.array(
                        [0.0] + [0.1]*(X_train_scaled.shape[1] - 1)),
                    eta=0.1, cost_function=cost_OLS,
                    n_epochs=n_epochs, batch_size=10,
                    gamma=0, scale_learning=True)
        print(
            f'Test {i+1}: MSE: {cost_OLS(beta_SGD, X_train_scaled, z_train):.6f}')
    
    print("No algorithm")
    for i in range(5):
        beta_SGD, num = SGD(
                    X=X_train_scaled, y=z_train_scaled,
                    theta_init=np.array(
                        [0.0] + [0.1]*(X_train_scaled.shape[1] - 1)),
                    eta=0.1, cost_function=cost_OLS,
                    n_epochs=n_epochs, batch_size=10,
                    gamma=0, scale_learning=False)
        print(
            f'Test {i+1}: MSE: {cost_OLS(beta_SGD, X_train_scaled, z_train):.6f}')


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
    X_test = helper.create_design_matrix(x_test, y_test, degree)
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    X_test_scaled = X_test - np.mean(X_train, axis=0)
    z_train_scale = np.mean(z_train, axis=0)
    z_train_scaled = z_train - z_train_scale

    # Analytical betas
    _, _, beta_OLS = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=degree, regression_method='OLS'
    )

    print(
        f'MSE (training): {helper.mean_squared_error(X_train_scaled @ beta_OLS + z_train_scale, z_train)} ANALYTICAL')
    print(
        f'MSE (testing): {helper.mean_squared_error(X_test_scaled @ beta_OLS + z_train_scale, z_test)} ANALYTICAL')
    print(
        f'R2-score (training): {helper.r2_score(X_train_scaled @ beta_OLS + z_train_scale, z_train)} ANALYTICAL')
    print(
        f'R2-score (testing): {helper.r2_score(X_test_scaled @ beta_OLS + z_train_scale, z_test)} ANALYTICAL')

    learning_rates = np.logspace(-1, -5, 5)
    train_R2_score = np.zeros(
        (len(list_no_of_minibatches), len(learning_rates)))
    test_R2_score = np.zeros(
        (len(list_no_of_minibatches), len(learning_rates)))

    # Looping through different learning rates
    for i, no_of_minibatches in enumerate(list_no_of_minibatches):
        batch_size = int(X_train.shape[0]/no_of_minibatches)
        updated_epochs = n_epochs * batch_size
        for j, eta in enumerate(learning_rates):
            beta_SGD, num = SGD(
                X=X_train_scaled, y=z_train_scaled,
                theta_init=np.array(
                    [0.0] + [0.01]*(X_train_scaled.shape[1] - 1)),
                eta=eta, cost_function=cost_OLS,
                n_epochs=updated_epochs, batch_size=batch_size,
                gamma=gamma)

            z_pred_train = (X_train_scaled @ beta_SGD) + z_train_scale
            z_pred_test = (X_test_scaled @ beta_SGD) + z_train_scale

            train_R2_score[i][j] = helper.r2_score(
                z_pred_train, z_train)
            test_R2_score[i][j] = helper.r2_score(
                z_pred_test, z_test)

    helper.seaborn_plot_batchsize_eta(
        score=train_R2_score,
        x_tics=learning_rates,
        y_tics=list_no_of_minibatches,
        score_name='Training R2-score',
        save_name=f'plots/test1/test1_OLS_gamma_{gamma}_epochs_{n_epochs}_training_12.png'
    )

    helper.seaborn_plot_batchsize_eta(
        score=test_R2_score,
        x_tics=learning_rates,
        y_tics=list_no_of_minibatches,
        score_name='Test R2-score',
        save_name=f'plots/test1/test1_OLS_gamma_{gamma}_epochs_{n_epochs}_test_12.png'
    )


def main_RIDGE(x_values, y_values, z_values, no_of_minibatches=10, n_epochs=200, degree=4, gamma=0):
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

    # TODO: input variables for lmbda values, eta values
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

    batch_sizes = [1, 5, 10, 30, 60, 100]
    for batch_size in batch_sizes:
        beta_SGD, num = SGD(
            X=X_train_scaled, y=z_train_scaled,
            theta_init=np.array(
                [0.0] + [0.1]*(X_train_scaled.shape[1] - 1)),
            eta=1e-1, cost_function=cost_RIDGE,
            n_epochs=batch_size, batch_size=batch_size,
            gamma=0.5,lmbda=1e-3)

        print(
            f'MSE: {cost_RIDGE(beta_SGD, X_train_scaled, z_train, 1e-3):.6f} (NUMERICAL (batch size = {batch_size}))')
    print('--------------------------------')
    
    # TODO: do some cleaning inside here
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            _, _, beta_RIDGE = helper.predict_output(
                x_train=x_train, y_train=y_train, z_train=z_train,
                x_test=x_test, y_test=y_test,
                degree=degree, regression_method='RIDGE', lmbda=lmbda
            )

            theta, num = SGD(
                theta_init=np.array(
                [0.0] + [0.1]*(X_train_scaled.shape[1] - 1)),
                eta=eta,
                cost_function=cost_RIDGE,
                n_epochs=n_epochs,
                batch_size=batch_size,
                X=X_train_scaled,
                y=z_train_scaled,
                gamma=gamma,
                lmbda=lmbda
            )

            z_pred_train = X_train_scaled @ theta + np.mean(z_train, axis=0)
            z_pred_test = X_test_scaled @ theta + np.mean(z_train, axis=0)

            train_R2_score[i][j] = helper.r2_score(z_train, z_pred_train)
            test_R2_score[i][j] = helper.r2_score(z_test, z_pred_test)

    fig, ax = plt.subplots(figsize=(8, 8))
    heat = sns.heatmap(train_R2_score, annot=True, ax=ax, cmap="viridis",
                       xticklabels=lmbda_values, yticklabels=learning_rates)

    # TODO: remove the save fig
    ax.set_title("Training R2-score")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig(
        f"plots/test1/test1_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_numdata_{len(x_values)}_noOfMinibatches_{no_of_minibatches}_degree_{degree}_training_1.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(test_R2_score, annot=True, ax=ax, cmap="viridis",
                xticklabels=lmbda_values, yticklabels=learning_rates)
    ax.set_title("Test R2-score")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig(
        f"plots/test1/test1_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_numdata_{len(x_values)}_noOfMinibatches_{no_of_minibatches}_degree_{degree}_test_1.png")
    plt.show()



if __name__ == '__main__':
    n = 500
    noise = 0.1
    x_values, y_values, z_values = helper.generate_data(n, noise)

    main_OLS(x_values, y_values, z_values)
    # main_RIDGE(x_values, y_values, z_values)
