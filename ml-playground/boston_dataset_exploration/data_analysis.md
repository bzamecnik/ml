# Boston house prices - data analysis

[data_analysis_report.txt](http://i.neural.cz/boston-dataset-exploration/data_analysis_report.txt)

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

![attribute correlations](http://i.neural.cz/boston-dataset-exploration/attr_correlations.png)

```
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
```

## Visualize Data

### Attribute Histograms

- What families of distributions are shown (if any)?
	- CRIM - after log it looks like two gaussians
	- CHAS - Bernoulli
	- RM - normal
	- MEDV - looks like normal
	- B, CRIM, RAD, ZN - a very strong peak on a side
	- which could be transformed?
		- CRIM - log
		- DIS - log
		- LSTAT - log
		- RM - ^2

![CRIM](http://i.neural.cz/boston-dataset-exploration/dist_CRIM.png)

![RM](http://i.neural.cz/boston-dataset-exploration/dist_RM.png)

![MEDV](http://i.neural.cz/boston-dataset-exploration/dist_MEDV.png)

![RAD](http://i.neural.cz/boston-dataset-exploration/dist_RAD.png)

![DIS](http://i.neural.cz/boston-dataset-exploration/dist_DIS.png)

![ZN](http://i.neural.cz/boston-dataset-exploration/dist_ZN.png)

When transformed (scatter with traget attribute):

![MEDV x log(CRIM)](http://i.neural.cz/boston-dataset-exploration/more/pair_CRIM_log.png)
![MEDV x log(DIS)](http://i.neural.cz/boston-dataset-exploration/more/pair_DIS_log.png)
![MEDV x log(LSTAT)](http://i.neural.cz/boston-dataset-exploration/more/pair_LSTAT_log.png)
![MEDV x log(RAD)](http://i.neural.cz/boston-dataset-exploration/more/pair_RAD_log.png)

All distributions: [AGE](http://i.neural.cz/boston-dataset-exploration/dist_AGE.png),
[B](http://i.neural.cz/boston-dataset-exploration/dist_B.png),
[CHAS](http://i.neural.cz/boston-dataset-exploration/dist_CHAS.png),
[CRIM](http://i.neural.cz/boston-dataset-exploration/dist_CRIM.png),
[DIS](http://i.neural.cz/boston-dataset-exploration/dist_DIS.png),
[INDUS](http://i.neural.cz/boston-dataset-exploration/dist_INDUS.png),
[LSTAT](http://i.neural.cz/boston-dataset-exploration/dist_LSTAT.png),
[MEDV](http://i.neural.cz/boston-dataset-exploration/dist_MEDV.png),
[NOX](http://i.neural.cz/boston-dataset-exploration/dist_NOX.png),
[PTRATIO](http://i.neural.cz/boston-dataset-exploration/dist_PTRATIO.png),
[RAD](http://i.neural.cz/boston-dataset-exploration/dist_RAD.png),
[RM](http://i.neural.cz/boston-dataset-exploration/dist_RM.png),
[TAX](http://i.neural.cz/boston-dataset-exploration/dist_TAX.png),
[ZN](http://i.neural.cz/boston-dataset-exploration/dist_ZN.png)

- Are there any obvious structures in the attributes that map to class values?

### Pairwise Scatter-plots

- What interesting two-dimensional structures are shown?
- What interesting relationships between the attributes to class values are shown?

<a href="http://i.neural.cz/boston-dataset-exploration/pairwise_scatter_matrix.png"
title="view/download original big image">
<img alt="pairwise scatter plot matrix"
src="http://i.neural.cz/boston-dataset-exploration/pairwise_scatter_matrix_t.png">
</a>

Some interesting joint plots:

![MEDV-LSTAT](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_LSTAT.png)

![MEDV-DIS](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_DIS.png)

![MEDV-AGE](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_AGE.png)

![INDUS-NOX](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_NOX.png)

![AGE-NOX](http://i.neural.cz/boston-dataset-exploration/joint_AGE_NOX.png)

![LSTAT-AGE](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_AGE.png)

![DIS-NOX](http://i.neural.cz/boston-dataset-exploration/joint_DIS_NOX.png)

![LSTAT-NOX](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_NOX.png)

All joint plots:

[`AGE_B`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_B.png),
[`AGE_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_CRIM.png),
[`AGE_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_DIS.png),
[`AGE_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_INDUS.png),
[`AGE_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_LSTAT.png),
[`AGE_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_MEDV.png),
[`AGE_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_NOX.png),
[`AGE_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_PTRATIO.png),
[`AGE_RM`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_RM.png),
[`AGE_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_TAX.png),
[`AGE_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_AGE_ZN.png)

[`B_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_B_AGE.png),
[`B_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_B_CRIM.png),
[`B_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_B_DIS.png),
[`B_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_B_INDUS.png),
[`B_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_B_LSTAT.png),
[`B_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_B_MEDV.png),
[`B_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_B_NOX.png),
[`B_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_B_PTRATIO.png),
[`B_RM`](http://i.neural.cz/boston-dataset-exploration/joint_B_RM.png),
[`B_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_B_TAX.png),
[`B_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_B_ZN.png)

[`CRIM_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_AGE.png),
[`CRIM_B`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_B.png),
[`CRIM_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_DIS.png),
[`CRIM_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_INDUS.png),
[`CRIM_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_LSTAT.png),
[`CRIM_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_MEDV.png),
[`CRIM_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_NOX.png),
[`CRIM_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_PTRATIO.png),
[`CRIM_RM`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_RM.png),
[`CRIM_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_TAX.png),
[`CRIM_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_CRIM_ZN.png)

[`DIS_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_AGE.png),
[`DIS_B`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_B.png),
[`DIS_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_CRIM.png),
[`DIS_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_INDUS.png),
[`DIS_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_LSTAT.png),
[`DIS_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_MEDV.png),
[`DIS_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_NOX.png),
[`DIS_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_PTRATIO.png),
[`DIS_RM`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_RM.png),
[`DIS_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_TAX.png),
[`DIS_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_DIS_ZN.png)

[`INDUS_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_AGE.png),
[`INDUS_B`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_B.png),
[`INDUS_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_CRIM.png),
[`INDUS_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_DIS.png),
[`INDUS_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_LSTAT.png),
[`INDUS_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_MEDV.png),
[`INDUS_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_NOX.png),
[`INDUS_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_PTRATIO.png),
[`INDUS_RM`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_RM.png),
[`INDUS_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_TAX.png),
[`INDUS_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_INDUS_ZN.png)

[`LSTAT_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_AGE.png),
[`LSTAT_B`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_B.png),
[`LSTAT_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_CRIM.png),
[`LSTAT_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_DIS.png),
[`LSTAT_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_INDUS.png),
[`LSTAT_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_MEDV.png),
[`LSTAT_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_NOX.png),
[`LSTAT_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_PTRATIO.png),
[`LSTAT_RM`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_RM.png),
[`LSTAT_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_TAX.png),
[`LSTAT_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_LSTAT_ZN.png)

[`MEDV_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_AGE.png),
[`MEDV_B`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_B.png),
[`MEDV_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_CRIM.png),
[`MEDV_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_DIS.png),
[`MEDV_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_INDUS.png),
[`MEDV_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_LSTAT.png),
[`MEDV_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_NOX.png),
[`MEDV_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_PTRATIO.png),
[`MEDV_RM`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_RM.png),
[`MEDV_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_TAX.png),
[`MEDV_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_MEDV_ZN.png)

[`NOX_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_AGE.png),
[`NOX_B`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_B.png),
[`NOX_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_CRIM.png),
[`NOX_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_DIS.png),
[`NOX_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_INDUS.png),
[`NOX_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_LSTAT.png),
[`NOX_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_MEDV.png),
[`NOX_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_PTRATIO.png),
[`NOX_RM`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_RM.png),
[`NOX_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_TAX.png),
[`NOX_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_NOX_ZN.png)

[`PTRATIO_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_AGE.png),
[`PTRATIO_B`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_B.png),
[`PTRATIO_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_CRIM.png),
[`PTRATIO_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_DIS.png),
[`PTRATIO_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_INDUS.png),
[`PTRATIO_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_LSTAT.png),
[`PTRATIO_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_MEDV.png),
[`PTRATIO_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_NOX.png),
[`PTRATIO_RM`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_RM.png),
[`PTRATIO_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_TAX.png),
[`PTRATIO_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_PTRATIO_ZN.png)

[`RM_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_RM_AGE.png),
[`RM_B`](http://i.neural.cz/boston-dataset-exploration/joint_RM_B.png),
[`RM_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_RM_CRIM.png),
[`RM_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_RM_DIS.png),
[`RM_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_RM_INDUS.png),
[`RM_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_RM_LSTAT.png),
[`RM_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_RM_MEDV.png),
[`RM_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_RM_NOX.png),
[`RM_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_RM_PTRATIO.png),
[`RM_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_RM_TAX.png),
[`RM_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_RM_ZN.png)

[`TAX_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_AGE.png),
[`TAX_B`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_B.png),
[`TAX_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_CRIM.png),
[`TAX_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_DIS.png),
[`TAX_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_INDUS.png),
[`TAX_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_LSTAT.png),
[`TAX_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_MEDV.png),
[`TAX_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_NOX.png),
[`TAX_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_PTRATIO.png),
[`TAX_RM`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_RM.png),
[`TAX_ZN`](http://i.neural.cz/boston-dataset-exploration/joint_TAX_ZN.png)

[`ZN_AGE`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_AGE.png),
[`ZN_B`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_B.png),
[`ZN_CRIM`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_CRIM.png),
[`ZN_DIS`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_DIS.png),
[`ZN_INDUS`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_INDUS.png),
[`ZN_LSTAT`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_LSTAT.png),
[`ZN_MEDV`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_MEDV.png),
[`ZN_NOX`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_NOX.png),
[`ZN_PTRATIO`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_PTRATIO.png),
[`ZN_RM`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_RM.png),
[`ZN_TAX`](http://i.neural.cz/boston-dataset-exploration/joint_ZN_TAX.png)
