# TODO: slett denne
# TODO: excercise 1
def plot_polynomial(x_true, y_true, z_true, beta):
    """
    # TODO: docstrings
    """
    fig = plt.figure()
    fig1 = plt.figure()
    ax1 = fig.gca(projection='3d')
    ax = fig.gca(projection='3d')
    # Make data.

    beta, X = get_betas(x_true, y_true, z_true)

    x_pent = np.linspace(0, 1, 100)
    y_pent = np.linspace(0, 1, 100)
    x_mesh, y_mesh = np.meshgrid(x_true, y_true)
    z_true = franke_function(x_mesh, y_mesh)
    #z_mesh = find_z_approx(x_mesh, y_mesh, z_true)

    print("c")

    z_approx = X @ beta

    # Plot the surface.
    surf = ax.plot_surface(x_mesh, y_mesh, z_true, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    surf1 = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    # # makeapproximated plot:
    # for i in range(len(x)):

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax1.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('% .02f'))
    ax1.zaxis.set_major_formatter(FormatStrFormatter('% .02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
    print("helooo")
    plt.show()
