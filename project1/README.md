# Project 1

## Report - folder ๐๐

#### project_1_FYS_STK4155.pdf ๐๐

## Code - folder ๐๐ฌ
For simplicity, we have created 6 files for the 6 different exercises. Here are some small explanation of the python-files included in this folder.

#### test_project_1.py ๐๐งช
Function for testing the different exercises. The file contains an easy way to reproduce **exerciseX.py**, with the preferred parameters. We have created the exercises chronologically, so hopefully there will be no surprises. 

##### *How to use this file properly:*
- Set the preffered parameters (all parameters are located in the beggining of the code)
- To test an exercise, set the belonging if-statement to True

#### helper.py ๐๐ฎ๐ผโโ๏ธ
This is our helper function. The functions that are included in this file are used in several exercises, and we chose to create this to improve code structure.

#### exercise1.py ๐๐
- Performing an OLS analysis using polynomials in x and y
- Find the confidence intervals of the parameters beta (printing them out in nice format)
- Evaluating the Mean Squared error (MSE)
- Evaluating the R^2 score

#### exercise2.py ๐๐ข
- Want to plot MSE vs complexity, with both the training and testing data
- Perform a bias-variance analysis of the Franke function, MSE vs complexity

#### exercise3.py ๐๐ผโ
- Compare the MSE you get from cross-validation with the one from bootstrap (print out)

#### exercise4.py ๐๐ (RIDGE)
- Perform the same bootstrap analysis (mse vs complexity) as in exercise 2
- Perform cross-validation as in exercise 3 (different values of lmbda)
- Perform a bias-variance analysis (MSE vs lmbda)


#### exercise5.py ๐ค ๐ชข (LASSO)
- Perform the same bootstrap analysis (mse vs complexity) as in exercise 2
- Perform cross-validation as in exercise 3 (different values of lmbda)
- Perform a bias-variance analysis (MSE vs lmbda)

#### exercise6.py ๐พ๐งญ
- Will do exercise 1 - 5 again
- Look into the main function - and uncomment the exercise you wanna run
- We have currently restricted the images to be in some specific coordinates. Just go to the function "read_terrain_data" and replace the boolean inside the  if-statement to run the code with the fully image/ terrain.  
    
ENJOYโบ๏ธ




