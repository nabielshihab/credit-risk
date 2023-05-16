# The Business Problem and The Analytical Approaches
**The Goal**  
A lending company that we are partnering with wanted us to **predict credit risk.**

**Why is it important?**  
Because it could tell lenders and other creditors whether a loan will end up being fully paid or charged off. Therefore, these predictions would be very useful to investors in determining whether or not to invest.

**The Approaches**  
As the goal requires us to make some predictions, we need to approach it using **predictive analytics**. In other words, we are going to deal with a machine learning model, or models(?) However, it does not mean that we will not take other approaches into account. For example, we can still perform some descriptive analytics to help us gain knowledge about the data.

# The Data
Download: https://bit.ly/loandataset   
We have a CSV-formatted dataset. It comprises the funded loans data only since they did not provide us the rejected loans dataset. It has **466,285 samples (rows)** and **75 features (columns)**, has multiple types of data, and has many columns consist of null values. There are columns related to identification numbers, some are related to the borrower / co-borrowers' personal information, some are loan-specific, some columns related to the borrower / co-borrowers' credit / public records, and other columns.

# The Steps Involved
Here is the list of the steps involved in the notebook to achieve the goal.     

**1. Data Understanding**  
    In this step, we will load the data and make a data dictionary to understand the meaning of each column in the dataset.  
    
**2. Data Preparation**  
    At this point, we need to ensure that our data is clean and is in a suitable form so our logic is provided with usable data. We are going to handle some potential issues, such as unnecessary samples, duplicate rows (if any), unrelevant / unusable features (feature selection), incorrect data types, bad-formatted values, categorical variables, and incomplete yet useful features. In addition, we will also add new features in this step.

**3. Exploratory Data Analysis (EDA)**    
    We will carry out some univariate and multivariate analyses to get a better understanding about the dataset, to know the characteristics of the dataset, to help make predictions and assumptions about the data, to find the patterns of the dataset, to find any relationships in a dataset, etc.  
    
**4. Modeling**  
    Here, we will create, compare, and evaluate our machine learning models. Tree-based models (Decision Tree and Random Forest) will be created. In addition, precision and balanced accuracy score will be used to measure the models' performance. 

## References: 
Some excellent sources of information:  
- https://daniellecrumley.github.io/LendingClubProject/
- https://blog.dataiku.com/tree-based-models-how-they-work-in-plain-english#:~:text=We%E2%80%99ll%20explore%20three%20types%20of%20tree-based%20models%3A%201,%E2%80%9Censemble%E2%80%9D%20method%20which%20builds%20many%20decision%20trees%20sequentially.
- https://www.analyticsvidhya.com/blog/2016/04/tree-based-algorithms-complete-tutorial-scratch-in-python/#:~:text=Tree%20based%20algorithms%20empower%20predictive%20models%20with%20high,kind%20of%20problem%20at%20hand%20%28classification%20or%20regression%29.
- https://www.statology.org/balanced-accuracy/
- https://datagy.io/sklearn-gridsearchcv/
