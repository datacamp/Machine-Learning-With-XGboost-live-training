# **Machine Learning with XGboost**<br/>by **Lis Sulmont**

Live training sessions are designed to mimic the flow of how a real data scientist would address a problem or a task. As such, a session needs to have some “narrative” where learners are achieving stated learning objectives in the form of a real-life data science task or project. For example, a data visualization live session could be around analyzing a dataset and creating a report with a specific business objective in mind _(ex: analyzing and visualizing churn)_, a data cleaning live session could be about preparing a dataset for analysis etc ... 

As part of the 'Live training Spec' process, you will need to complete the following tasks:

Edit this README by filling in the information for steps 1 - 4.

## Step 1: Foundations 

This part of the 'Live training Spec' process is designed to help guide you through session design by having you think through several key questions. Please make sure to delete the examples provided here for you.

### A. What problem(s) will students learn how to solve? (minimum of 5 problems)

- How to decide whether to use gradient boosting or not
- How to instantiate and customize `XGBoost` models
- How to use `XGBoost`'s `DMatrix` to optimize computing performance
- How to evaluate models in `XGBoost` using the right metrics
- How to tune parameters in `XGBoost` to achieve the best results
- How to visualize trees in `XGBoost` to analyze feature importance

Note that there will be no pre-processing in this live training. The data will be presented with clean ready-to-use features.


### B. What technologies, packages, or functions will students use? Please be exhaustive.

- `pandas`
- `numpy`
- `scikit-learn` (maybe)
- `xgboost`

### C. What terms or jargon will you define?

_Whether during your opening and closing talk or your live training, you might have to define some terms and jargon to walk students through a problem you’re solving. Intuitive explanations using analogies are encouraged._

- **Decision Trees**: machine learning technique that uses tree structures. At each node, the data is split into pieces based on the value of a feature. The algorithm finds the split by finding the best information gain possible. It can be used for both regression and classification problem.
- **CART**: stands for *Classification and Regression trees*. Common way to refer to decision trees.
- **Leaf node**: node that has no children, or leads to no other nodes. At a leaf node, a decision is outputted.
- **Root node**: the top node of a tree.
- **Tree Depth**: the length of the longest path from the root node to a leaf node.
- **Boosting**
- **Bagging**
- **Gradient boosting (aka GBM and GBDT)**
- **Loss function**
- **Base learners**
- **Weak Learners**

### D. What mistakes or misconceptions do you expect? 

_To help minimize the amount of Q&As and make your live training re-usable, list out some mistakes and misconceptions you think students might encounter along the way._


- Confusing gradient boosting with gradient descent
- Confusing boosting and bagging
- Assuming that using XGBoost is always a good idea
- Assuming that `DMatrix` is the same as `numpy`'s `array` or `pandas`'s `data frames`
- `XGBoost` is a library specialized in gradient boosting. It is not an acronym or slang for gradient boosting. And, there are other libraries that allow you too implement gradient boostings (e.g, `scikit-learn`).

### E. What datasets will you use? 

Live training sessions are designed to walk students through something closer to a real-life data science workflow. Accordingly, the dataset needs to accommodate that user experience. 
As a rule of thumb, your dataset should always answer yes to the following question: 
> Is the dataset/problem I’m working on, something an industry data scientist/analyst could work on? 

Check our [datasets to avoid](https://instructor-support.datacamp.com/en/articles/2360699-datasets-to-avoid) list. 

**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand)

**Problem:** Predict whether a booking will be cancelled

## Step 2: Who is this session for?

Terms like "beginner" and "expert" mean different things to different people, so we use personas to help instructors clarify a live training's audience. When designing a specific live training, instructors should explain how it will or won't help these people, and what extra skills or prerequisite knowledge they are assuming their students have above and beyond what's included in the persona.

- [X] Please select the roles and industries that align with your live training. 
- [X] Include an explanation describing your reasoning and any other relevant information. 

### What roles would this live training be suitable for?

*Check all that apply.*

- [ ] Data Consumer
- [ ] Leader 
- [X] Data Analyst
- [X] Citizen Data Scientist
- [X] Data Scientist
- [ ] Data Engineer
- [ ] Database Administrator
- [ ] Statistician
- [X] Machine Learning Scientist
- [ ] Programmer
- [ ] Other (please describe)

XGBoost is a powerful machine learning library that became very popular after winning several Kaggle competitions. It is an asset to anyone in machine learning or looking to upskill in machine learning (e.g., data analysts and citizen data scientists).

### What industries would this apply to?

*List one or more industries that the content would be appropriate for.*

This is relevant for all industries. 

This isn't an industry, but this course is especially interesting for anyone who likes doing Kaggle competitions.


### What level of expertise should learners have before beginning the live training?

*List three or more examples of skills that you expect learners to have before beginning the live training*

> - Can draw common plot types (scatter, bar, histogram) using matplotlib and interpret them
> - Can run a linear regression, use it to make predictions, and interpret the coefficients.
> - Can calculate grouped summary statistics using SELECT queries with GROUP BY clauses.

- Can confidently use `scikit-learn` to train machine learning models, including functions like `fit()`, `predict()`, and `train_test_split()`.
- Can describe a decision tree (or CART)
- Can train and tune decision tree and random forest models in `scikit-learn` with cross validation.
- Can define hyperparameter tuning, underfitting, overfitting, the bias-variance tradeoff, regularization, classification, regression, cross validation, grid search, and random search. These are concepts covered in [Supervised Learning with scikit-learn](https://learn.datacamp.com/courses/supervised-learning-with-scikit-learn), which is a pre-req to [Machine Learning with Tree-Based Models in Python](https://learn.datacamp.com/courses/machine-learning-with-tree-based-models-in-python) (listed below).


## Step 3: Prerequisites

List any prerequisite courses you think your live training could use from. This could be the live session’s companion course or a course you think students should take before the session. Prerequisites act as a guiding principle for your session and will set the topic framework, but you do not have to limit yourself in the live session to the syntax used in the prerequisite courses.

- [Machine Learning with Tree-Based Models in Python](https://learn.datacamp.com/courses/machine-learning-with-tree-based-models-in-python) (at least first three chapters): Course student should take before the session because students should be comfortable with decision trees and have heard of the concept of ensemble methods with tree-based models (bagging, random forest). Otherwise, I will have to spend too much time introducing these topics.
-  [Extreme Gradient Boosting with XGBoost](https://learn.datacamp.com/courses/extreme-gradient-boosting-with-xgboost): Companion course, since it focuses on implementation with the XGBoost library. Other courses, including the one above, use the scikit-learn implementation.


## Step 4: Session Outline

A live training session usually begins with an introductory presentation, followed by the live training itself, and an ending presentation. Your live session is expected to be around 2h30m-3h long (including Q&A) with a hard-limit at 3h30m. You can check out our live training content guidelines [here](_LINK_). 



### Introduction Slides 
- Introduction to the webinar and instructor (led by DataCamp TA)
- History of gradient boosting going from decision trees, bagging, random Forrests, to boosting. Terms will be explained on top of each other. It will end with an understanding of gradient boosting.
- Why do you want to use XGBoost? 
	- Compare its performance with other ML libraries 
	- Introduce `XGBoost`'s `DMatrix` as an alternative to dataframes
- Quick overview of the parameters available in gradient boosting
- Introduction of problem: can we predict whether a hotel booking will be cancelled?
- Set expectations about Q&A

### Live Training
#### Your first XGBoost Model
- Import data and print header of DataFrame `pd.read_excel()`, `.head()`
- Glimpse at the data to
  - Get column types using `.dtypes`
  - Use `.describe()`, `.info()`
  - **Q&A** 

#### Digging into Parameters
- Convert date columns to datetime `pd.to_datetime()`
- Change column names
- Extract year, month from datetime `.strftime()`
- Drop an irrelevant column `.drop()`
- Fill missing values with `.fillna()`

#### Hyperparameter tuning
- Use `XGBoost`'s [Scikit-Learn API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
- Grid search with `scikit-learn`'s `GridSearchCV`
- Random search with `scikit-learn`'s `RandomizedSearchCV`

### Ending slides
- Recap of what we learned
- What are the limits of gradient boosting? When should it not be used?
- Call to action and course recommendations

## Authoring your session

To get yourself started with setting up your live session, follow the steps below:

1. Download and install the "Open in Colabs" extension from [here](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo?hl=en). This will let you take any jupyter notebook you see in a GitHub repository and open it as a **temporary** Colabs link.
2. Upload your dataset(s) to the `data` folder.
3. Upload your images, gifs, or any other assets you want to use in the notebook in the `assets` folder.
4. Check out the notebooks templates in the `notebooks` folder, and keep the template you want for your session while deleting all remaining ones.
5. Preview your desired notebook, press on "Open in Colabs" extension - and start developing your content in colabs _(which will act as the solution code to the session)_.  :warning: **Important** :warning: Your progress will **not** be saved on Google Colabs since it's a temporary link. To save your progress, make sure to press on `File`, `Save a copy in GitHub` and follow remaining prompts. You can also download the notebook locally and develop the content there as long you test out that the syntax works on Colabs as well.
6. Once your notebooks is ready to go, give it the name `session_name_solution.ipynb` create an empty version of the Notebook to be filled out by you and learners during the session, end the file name with `session_name_learners.ipynb`. 
7. Create Colabs links for both sessions and save them in notebooks :tada: 
