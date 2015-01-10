# Boston house prices - problem definition

An excercise inspired by the [Applied Machine Learning Process](http://machinelearningmastery.com/applied-machine-learning-process/) book applied on the *Boston house prices* dataset.

## What is the problem?

### Informal description

> Estimate the price of a house.

### Formal description

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

- Task (T): Estimate the price of a house given its attributes (regression).
- Experience (E): Attributes and prices of houses sold in past.
- Performance (P): Difference between predicted and actual price (for one instance), R^2 score for a bunch of houses (how the prediction is better than guessing with average price).

### Assumptions

- attributes that may correlate with price positively:
	- RAD, RM, ZN
- attributes that may correlate with price negatively:
    - AGE, B, CRIM, CRIM, DIS, INDUS, LSTAT, NOX, PTRATIO, TAX

### Similar problems

- What constraints were imposed on the collection of the data?
- When and from where was the data collected?

http://archive.ics.uci.edu/ml/datasets/Housing

[Hedonic
prices and the demand for clean air](http://deepblue.lib.umich.edu/bitstream/handle/2027.42/22636/0000186.pdf?sequence=1&isAllowed=y), Harrison, D. and Rubinfeld, D.L., J. Environ. Economics & Management,
vol.5, 81-102, 1978.

Data is from 1970s, ie. cca 40 years old in 2014.

Raw data:

[housing.data](http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data)
[housing.names](http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)

Data available via the scikit-learn API:

```
from sklearn.datasets import load_boston
load_boston()

```

### Description of provided data

- How many attributes and instances are there?
- What do each of the attributes mean in the context of the problem domain?

Number of instances: 506

Attributes (13):

- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population

Target attribute:

- MEDV     Median value of owner-occupied homes in $1000's

## Why does the problem need to be solved?

### Motivation

It is an excercise in ML - problem overview and data preparation.

In real world it would be useful both for real estate agencies, house owners willing to sell the house or potential buyers.

### Benefits

Explore the various phases of the ML process, in this case mainly problem definition and data preparation.

In real world: estimating prices more precisely than human guess -> higher liquidity -> more houses sold -> useful for all parties interested.

### Use

To have a template for solving further ML problems.

## How would I solve the problem (manually)?

I'd look at the data, compare each attribute to houses with similar attribute values and try to average the price somehow.
