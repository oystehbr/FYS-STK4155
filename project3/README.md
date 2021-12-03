# Project 3

## Report - folder 📒📖

#### project_3_FYS_STK4155.pdf 📈📝

## Code - folder 🔐💬
We have added the python files to this folder. One for testing all the features of our code, and to be able to verify the results we are 
getting in the report. Two files for collecting all activation functions and cost-functions used in the project. We have also re-used the  gradient descent method from last project. Last one is a "little" helper file, which contains several functions we are using often (copied from project2, and added some new ones).

We have used tensorflow's neural network and scikitlearn's decision tree and random forest algorithms. 

#### >> test_project_3.py 🎓🧪
Function for testing results that we present in the report. The file contains an easy way to reproduce **test X**, with the preferred parameters. The preffered parameters may or may not be those we are refering to in the report. Some of the values may or may not need to be changed for the results to be the same as we are getting in the report. 

##### *How to use this file properly:*
- Close all if-statements (VSCODE: CTRL K + CTRL 0), to have a nice scrolling time.
- Scroll to the test you want to run, set the testX parameter to True. 
- Small description is provided before the testX boolean.
- It is easier to look at one test at a time, therefore I recommend you to turn one test to True at a time.
- If you want to test the code of some results in the report, you may need to change the parameters inside testX

#### >> helper.py 🚑👮🏼‍♂️
This is our helper function. The functions that are included in this file are used in several times in the project, and we chose to create this to improve code structure. For instance, we are loading the data in this file, where we receive the training and testing data by calling the functions. 

#### >> activation_functions.py 🤼‍♂️🎚
- As the filename highlights, collection of the different activation we are using in this project 

#### >> cost_functions.py 💰💸
- As the filename highlights, collection of the different cost-functions we are using in this project

#### >> gradient_descent.py 🏔📉
Includes three functions:
- **SGD**-function: do stochastic gradient descent and optimize against a cost-function that will be provided into the function. Used in the logistic regression algorithm in this project. 

#### >> requirements.txt 🔞
We have added a requirements file, including all the packages needed for this project.

#### data (folder) 🖥📜
The data sets we are using in the project and some more we have been testing. Those are of type xlsx and csv. 

#### not_used (folder) ⛔️🗑
Our own created neural network class. Now, with a new feature: it can predict multiclasses (so it is possible to have multiple output nodes). We are not using this class in this project



ENJOY☺️


