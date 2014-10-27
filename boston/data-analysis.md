# Boston house prices - data analysis

## Summarize Data

### Data Structure

- How many attributes and instances are there?
- What are the data types of each attribute (e.g. nominal, ordinal, integer, real, etc.)?

Number of instances: 506

Attributes (13):

- CRIM: real - per capita crime rate by town
- ZN: real - proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: real - proportion of non-retail business acres per town
- CHAS: int (category) - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX: real - nitric oxides concentration (parts per 10 million)
- RM: real - average number of rooms per dwelling
- AGE: real - proportion of owner-occupied units built prior to 1940
- DIS: real - weighted distances to five Boston employment centres
- RAD: int (category) - index of accessibility to radial highways
- TAX: int (could be real) - full-value property-tax rate per $10,000
- PTRATIO: rea - pupil-teacher ratio by town
- B: real - function of proportion of blacks by town (1000(Bk - 0.63)^2)
- LSTAT: real - percentage of lower social status population

Target attribute:

- MEDV: real - median value of owner-occupied homes in $1000's

### Data distributions

```
       count        mean         std        min        50%       max       mode
CRIM     506    3.593761    8.596783    0.00632    0.25651   88.9762    0.01501
ZN       506   11.363636   23.322453    0.00000    0.00000  100.0000    0.00000
INDUS    506   11.136779    6.860353    0.46000    9.69000   27.7400   18.10000
CHAS     506    0.069170    0.253994    0.00000    0.00000    1.0000    0.00000
NOX      506    0.554695    0.115878    0.38500    0.53800    0.8710    0.53800
RM       506    6.284634    0.702617    3.56100    6.20850    8.7800    5.71300
AGE      506   68.574901   28.148861    2.90000   77.50000  100.0000  100.00000
DIS      506    3.795043    2.105710    1.12960    3.20745   12.1265    3.49520
RAD      506    9.549407    8.707259    1.00000    5.00000   24.0000   24.00000
TAX      506  408.237154  168.537116  187.00000  330.00000  711.0000  666.00000
PTRATIO  506   18.455534    2.164946   12.60000   19.05000   22.0000   20.20000
B        506  356.674032   91.294864    0.32000  391.44000  396.9000  396.90000
LSTAT    506   12.653063    7.141062    1.73000   11.36000   37.9700    6.36000
MEDV     506   22.532806    9.197104    5.00000   21.20000   50.0000   50.00000
```

Missing values: none

#### Correlations with target variable

Pearson correlations with absolute value > 0.5:

- RM: 0.695360
- PTRATIO: -0.507787
- LSTAT: -0.737663

The average number of rooms is likely to correlate with price, its no surprise. The social status and availability of education is also likely to be correlated, still it might not be sure at the first sight.

#### Correlations among attributes (redundancy)

Intra-attribute (Pearson) correlations with absolute value > 0.5.

(RAD, TAX)     0.910228
(INDUS, NOX)   0.763651
(AGE, NOX)     0.731470
(INDUS, TAX)   0.720760
(NOX, TAX)     0.668023
(DIS, ZN)      0.664408
(AGE, INDUS)   0.644779
(CRIM, RAD)    0.622029
(NOX, RAD)     0.611441
(INDUS, LSTAT) 0.603800
(AGE, LSTAT)   0.602339
(INDUS, RAD)   0.595129
(LSTAT, NOX)   0.590879
(CRIM, TAX)    0.579564
(LSTAT, TAX)   0.543993
(AGE, TAX)     0.506456
(NOX, ZN)     -0.516604
(INDUS, ZN)   -0.533828
(DIS, TAX)    -0.534432
(AGE, ZN)     -0.569537
(LSTAT, RM)   -0.613808
(DIS, INDUS)  -0.708027
(AGE, DIS)    -0.747881
(DIS, NOX)    -0.769230

## Visualize Data

### Attribute Histograms

- What families of distributions are shown (if any)?
- Are there any obvious structures in the attributes that map to class values?

TODO

### Pairwise Scatter-plots

- What interesting two-dimensional structures are shown?
- What interesting relationships between the attributes to class values are shown?

TODO
