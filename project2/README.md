# Project 2

## Report - folder 📒📖

#### project_1_FYS_STK4155.pdf 📈📝

## Code - folder 🔐💬
We have added the python files to this folder. One for testing all the features of our code, and to be able to verify the results we are 
getting in the report. Two files for collecting all activation functions and cost-functions used in the project. Then we have our main-files, that is a file including the Neural Netork class and the gradient descent method. Last one is a "little" helper file, which contains several functions we are using often (copied from project1, and added some new ones).

#### >> test_project_2.py 🎓🧪
Function for testing results that we present in the report. The file contains an easy way to reproduce **test X**, with the preferred parameters. The preffered parameters may or may not be those we are refering to in the report. Some of the values may or may not need to be changed for the results to be the same as we are getting in the report.

##### *How to use this file properly:*
- Close all if-statements (VSCODE: CTRL K + CTRL 0), to have a nice scrolling time.
- Scroll to the test you want to run, set the testX parameter to True. 
- Small description is provided before the testX boolean.
- It is easier to look at one test at a time, therefore I recommend you to turn one test to True at a time.
- If you want to test the code of some results in the report, you may need to change the parameters inside testX

#### >> helper.py 🚑👮🏼‍♂️
This is our helper function. The functions that are included in this file are used in several times in the project, and we chose to create this to improve code structure. 

#### >> activation_functions.py 🤼‍♂️🎚
- As the filename highlights, collection of the different activation we are using in this project 

#### >> cost_functions.py 💰💸
- As the filename highlights, collection of the different cost-functions we are using in this project

#### >> gradient_descent.py 🏔📈
Includes three functions:
- **SGD**-function: do stochastic gradient descent and optimize against a cost-function that will be provided into the function.
- **main_OLS**-function: function for get result for finding the optimal learning rate and number of mini-batches, provides a R2-score (seaborn) plot.
- **main_RIDGE**-function: using stochastic gradient descent to find optimal hyperparameter lambda and learning rate, by looking at some seaborn plot.

#### >> FF_Neural_Network.py 🔗🕸
Inside this file, we have created a Neural Network class. We have provided great docstrings to the different methods in the class, but we will give out some short information of them here. The class is able to create a neural network with various input nodes, hidden nodes and hidden layers. 

**Neural_Network**-class:

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




