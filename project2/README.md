# Project 2

## Report - folder 📒📖

#### project_1_FYS_STK4155.pdf 📈📝

## Code - folder 🔐💬
We have added # TODO files to this folder. One for testing all the features of our code, and to be able to verify the results we are 
getting in the report. Two files for collecting all activation functions and cost-functions used in the project. Then we have our main-files, that is a file including the Neural Netork class and the gradient descent method. Last one is a "little" helper file, which contains several functions we are using often (copied from project1, and added some new ones).

#### >> test_project_1.py 🎓🧪
Function for testing the different exercises. The file contains an easy way to reproduce **exerciseX.py**, with the preferred parameters. We have created the exercises chronologically, so hopefully there will be no surprises. 

##### *How to use this file properly:*
# TODO: write this better
- Set the preffered parameters (all parameters are located in the beggining of the code)
- To test an exercise, set the belonging if-statement to True

#### >> helper.py 🚑👮🏼‍♂️
This is our helper function. The functions that are included in this file are used in several times in the project, and we chose to create this to improve code structure. 

#### >> activation_functions.py 🤼‍♂️🎚
- As the filename highlights, collection of the different activation we are using in this project 

#### >> classification_problem.py 0️⃣1️⃣
- TODO: delete maybe -> just for testing, but this is also inside the test_project_2.py (it's just a case of the Neural Network)

#### >> cost_functions.py 💰💸
- As the filename highlights, collection of the different cost-functions we are using in this project

#### >> gradient_descent.py 🏔📈
Includes three functions:
- **SGD**-function: do stochastic gradient descent and optimize against a cost-function that will be provided into the function.
- **main_OLS**-function: function for comparing the gradient descent solution of the OLS-problem vs. the actual solution of the problem.
- **main_RIDGE**-function: using stochastic gradient descent to find optimal hyperparameter lambda and learning rate, by looking at some seaborn plot.

#### >> logistic_regression.py 🚜🚚
- TODO: maybe delete -> move stuff to activation-funciton.py -> maybe double. I don't know

#### >> FF_Neural_Network.py 🔗🕸
Inside this file, we have created a Neural Network class. We have provided great docstrings to the different methods that are included in the class, but will give out some small indications of the methods here.

- **\_\_init__**-method: initialize the Neural Network, initializes the weights according to the standard normal distribution and the biases with a value of 0.01 for every node. It will also set some default activation function for the hidden- and outputlayer, which can be reinitialized with a call on a different method inside the class. 
- **feed_forward**-method: will do the feed_forward concept in a Neural Network, with the current weights and biases it will "predict" the outcome given the input.
- **backpropagation**-method: will do the backpropagation concept in a Neural Network. It will give return some gradients which depends on the currenct weights and biases. 
- **SGD**-method: using the stochastic gradient descent on the weights and the biases of the Neural Network. 
- **plot_cost_of_last_training**-method: method for plotting the value of the cost-function (that the Neural Network is optimizing against), vs. the number of iterations in the SGD. This will only be able if you have provided some "keep"-boolean inside the *train_model*-method.
- **plot_accuracy_of_last_training**-method: method for plotting the accuracy vs. the number of iterations in the SGD. This will only be able if you have provided some "keep"-boolean inside the *train_model*-method.
- **initialize_the_biases**-method: method for initializing the biases, this will be called by the *__init__*-method, and whenever we want to refresh the biases (e.g. when we want to do a seaborn-plot with slightly different SGD values, but with a fresh model)
- **initialize_the_weights**-method: method for initializing the weights, this will be called by the *__init__*-method, and whenever we want to refresh the weights (e.g. when we want to do a seaborn-plot with slightly different SGD values, but with a fresh model)
- **set_SGD_values**-method: method for setting the preffered values that the stochastic gradient descent will be using when training the model. You can choose to either change one or several values. This function is called in the *__init__*-method to initialize some SGD values.
- **set_activation_function_hidden_layers**-method: method for setting the activation function for the hidden layer
- **set_activation_function_output_layers**-method: method for setting the activation function for the output layer
- **set_cost_function**-method: method for setting the cost function for the Neural Network to optimize against. It will try to train the model to achieve as low cost as possible.
- **train_model**-method: fancy name for calling the SGD method. 








ENJOY☺️




