Project Outline
===============

For the Coursera Practical Machine Learning final assignment I designed
and implemented a model using data from accelerometers placed on the
belt, forearm, arm, and dumbell of 6 participants of an experiment to
predict the manner in which they did the exercise.

Using the data collected in
<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>,
4 different predictive models (namely using Regression Tree, Random
Forest, Bagging and Boosting algorithm) have been built and compared
using a similar setting. Then, the model that resulted with higerh
accuracy was further improved and, finally, used on the test dataset.

The workflow of this project is basically:

-   [loading and filtering the data](#exploratory-data-analysis)
-   [creating a testing and validating
    test](#test-and-validation-dataset)
-   [building four different models and calculating their accuracy
    against the validation set](#predictive-models-for-the-dataset)
-   [choosing the final model](#choosing-the-model)
-   [using the final model on the provodided test
    set](#applying-the-model-to-the-test-set)

Exploratory data analysis
=========================

This section describes how the data has been loaded into R data.frame
structures and, from the raw database, the process for extracting the
features to build the predictive models.

Loading the data
----------------

The main source of the data is
<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>.
The data were downloaded in a local folder and then loaded in two
different datasets: `dataset` for the training and `testset` for the
test sets.

The datasets have the following size:

    ## [1] 19622   160

    ## [1]  20 160

The `dataset` data frame consists of 19622 datapoint of 160 recorded
features and the `testset` it’s just 20 data points. `testset` contains
20 unknown test cases to be predicted by the model.

Feature Extraction
------------------

The first step has been to check what is the percentage of available
data for each feature as the data.frame columns may contain not valid
elements. A `validelements` function has been created that, for each
data.frame column, count their valid elements (i.e. not empty, NaN or
`#DIV/0!`). The function has been used on the `dataset` data.frame.

<img src="PaoloCoraggioFinalAssignment_files/figure-markdown_strict/unnamed-chunk-3-1.png"  />

The plot shows that data.frame variables (features) contain or 100%
valide data or very few (less than 5% of valide data). The features
containing less the 5% of data will be discharged.

Moreover, the first 7 features contain temporal information that has
been chosen not to be considered in this project (a forcasting approach
would be more suitable).

    dataset <- dataset[, p.validelements > 0.05]
    testset <- testset[, p.validelements > 0.05]

    ## trimming the first 7 elements

    dataset <- dataset[-c(1:7)]
    testset <- testset[-c(1:7)]

The final data.frame now have the following sizes:

    ## [1] 19622    53

    ## [1] 20 53

As we can see, the dataset dimension, and so its complexity, has been
reduced making also more parsimonious its analisys.

Test and Validation Dataset
===========================

The `dataset` has been splitted in two, 70% to train the different
models and 30% for validating them.

    set.seed(123)

    inTrain <- createDataPartition(dataset$classe, p = 0.7, list = FALSE)
    train.set <- dataset[inTrain,]
    validation.set <- dataset[-inTrain,]

    dim(train.set)

    ## [1] 13737    53

    dim(validation.set)

    ## [1] 5885   53

The target variable for our analysis is the feature `classe` that shows
the following distribution.

<table>
<caption>Percentage of classes frequencies in the different datasets</caption>
<thead>
<tr class="header">
<th></th>
<th style="text-align: right;">Freq Dataset</th>
<th style="text-align: right;">%</th>
<th style="text-align: right;">Freq Training set</th>
<th style="text-align: right;">%</th>
<th style="text-align: right;">Freq Testing set</th>
<th style="text-align: right;">%</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>A</td>
<td style="text-align: right;">5580</td>
<td style="text-align: right;">28.4</td>
<td style="text-align: right;">3906</td>
<td style="text-align: right;">28.4</td>
<td style="text-align: right;">1674</td>
<td style="text-align: right;">28.4</td>
</tr>
<tr class="even">
<td>B</td>
<td style="text-align: right;">3797</td>
<td style="text-align: right;">19.4</td>
<td style="text-align: right;">2658</td>
<td style="text-align: right;">19.3</td>
<td style="text-align: right;">1139</td>
<td style="text-align: right;">19.4</td>
</tr>
<tr class="odd">
<td>C</td>
<td style="text-align: right;">3422</td>
<td style="text-align: right;">17.4</td>
<td style="text-align: right;">2396</td>
<td style="text-align: right;">17.4</td>
<td style="text-align: right;">1026</td>
<td style="text-align: right;">17.4</td>
</tr>
<tr class="even">
<td>D</td>
<td style="text-align: right;">3216</td>
<td style="text-align: right;">16.4</td>
<td style="text-align: right;">2252</td>
<td style="text-align: right;">16.4</td>
<td style="text-align: right;">964</td>
<td style="text-align: right;">16.4</td>
</tr>
<tr class="odd">
<td>E</td>
<td style="text-align: right;">3607</td>
<td style="text-align: right;">18.4</td>
<td style="text-align: right;">2525</td>
<td style="text-align: right;">18.4</td>
<td style="text-align: right;">1082</td>
<td style="text-align: right;">18.4</td>
</tr>
</tbody>
</table>

The class distribution is almost balanced except for the Class `A` that
contains a slight higher number of samples. As we can see the Class
distribuition has been preserved by the `createDataPartition` function.

Predictive models for the dataset
=================================

This section is about how 4 different predictive models have been
designed and implemented in order to chose the most promising one

Cross Validation
----------------

As we will compare different algorithms, a preset Cross Validation
parameter is set for all different models. Since the training dataset
contains a sufficient number of points, a basic cross validation choise
for this kind of dataset is 5-fold cross-validation to estimate
accuracy. In order to seek a better estimate, each algorithm will be
repeated 3 times on each folder.

    control <- trainControl(method = "repeatedcv", 
                            number = 5, 
                            repeats = 3,
                            verboseIter = TRUE)
    metric <- "Accuracy"

Classification Tree
-------------------

The first model is the simplest one: a classification tree. I am using
the `train` function from `caret` library using `rpart` method.

    set.seed(111)

    start.time <- Sys.time()
    mod.CT <- train(classe ~., 
                    data = train.set,
                    method = 'rpart',
                    tuneLength = 25,
                    trControl = control,
                    metric = metric)

    end.time <- Sys.time()
    time.takenCT <- end.time - start.time
    time.takenCT

    save(mod.CT, file = "modeCT.RData")
    save(time.takenCT, file = "timeCT.RData")

With an accuracy measured on the validation set

    confusionMatrix(predict(mod.CT,newdata = validation.set),
                    validation.set$classe)$overall

<table>
<thead>
<tr class="header">
<th style="text-align: right;">Accuracy</th>
<th style="text-align: right;">Kappa</th>
<th style="text-align: right;">AccuracyLower</th>
<th style="text-align: right;">AccuracyUpper</th>
<th style="text-align: right;">AccuracyNull</th>
<th style="text-align: right;">AccuracyPValue</th>
<th style="text-align: right;">McnemarPValue</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;">0.806</td>
<td style="text-align: right;">0.755</td>
<td style="text-align: right;">0.796</td>
<td style="text-align: right;">0.816</td>
<td style="text-align: right;">0.284</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
</tr>
</tbody>
</table>

Random Forest
-------------

    set.seed(111)

    start.time <- Sys.time()
    mod.RF <- train(classe ~.,
                    data = train.set,
                    method = 'rf',
                    trControl = control,
                    metric = metric,
                    tuneLength = 25)

    end.time <- Sys.time()
    timeRF.taken <- end.time - start.time
    timeRF.taken

    save(mod.RF, file = "modRF.RData")
    save(timeRF.taken, file = "timeRF.RData")

Gives the following accuracy:

    confusionMatrix(predict(mod.RF, validation.set), 
                    validation.set$classe)$overall

<table>
<thead>
<tr class="header">
<th style="text-align: right;">Accuracy</th>
<th style="text-align: right;">Kappa</th>
<th style="text-align: right;">AccuracyLower</th>
<th style="text-align: right;">AccuracyUpper</th>
<th style="text-align: right;">AccuracyNull</th>
<th style="text-align: right;">AccuracyPValue</th>
<th style="text-align: right;">McnemarPValue</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">0.996</td>
<td style="text-align: right;">0.284</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">NaN</td>
</tr>
</tbody>
</table>

Boosting
--------

    set.seed(111)

    start.time <- Sys.time()
    mod.Boosting <- train(classe ~.,
                          data = train.set,
                          method = "gbm",
                          trControl = control,
                          metric = metric,
                          verbose = FALSE)

    end.time <- Sys.time()
    time.Boosting <- end.time - start.time
    time.Boosting

With accuracy:

    round(confusionMatrix(predict(mod.Boosting, validation.set), 
                    validation.set$classe)$overall,3)

<table>
<thead>
<tr class="header">
<th style="text-align: right;">Accuracy</th>
<th style="text-align: right;">Kappa</th>
<th style="text-align: right;">AccuracyLower</th>
<th style="text-align: right;">AccuracyUpper</th>
<th style="text-align: right;">AccuracyNull</th>
<th style="text-align: right;">AccuracyPValue</th>
<th style="text-align: right;">McnemarPValue</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;">0.964</td>
<td style="text-align: right;">0.955</td>
<td style="text-align: right;">0.959</td>
<td style="text-align: right;">0.969</td>
<td style="text-align: right;">0.284</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
</tr>
</tbody>
</table>

Bagging
-------

    set.seed(111)

    start.time <- Sys.time()
    mod.Bagging <- train(classe ~.,
                         data = train.set,
                         method = "treebag",
                         trControl = control,
                         metric = metric)

    end.time <- Sys.time()
    time.Bagging <- end.time - start.time

    round(confusionMatrix(predict(mod.Bagging, validation.set), 
                validation.set$classe)$overall,3)

With an accuracy:

<table>
<thead>
<tr class="header">
<th style="text-align: right;">Accuracy</th>
<th style="text-align: right;">Kappa</th>
<th style="text-align: right;">AccuracyLower</th>
<th style="text-align: right;">AccuracyUpper</th>
<th style="text-align: right;">AccuracyNull</th>
<th style="text-align: right;">AccuracyPValue</th>
<th style="text-align: right;">McnemarPValue</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;">0.984</td>
<td style="text-align: right;">0.98</td>
<td style="text-align: right;">0.981</td>
<td style="text-align: right;">0.987</td>
<td style="text-align: right;">0.284</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.001</td>
</tr>
</tbody>
</table>

Choosing the model
==================

As we can see from the accurancy measured on the testing set, the Random
Forest approach looks preferible with respect the other ones. The
accuracy is already quite high (about 99.450%). The next model tries to
tune further the Random Forest model by estimating a more approrpiate
value for `ntree` and `mtry` parameters using a random search. We will
use the model to estimate the parameters’ tree as well.

    controlRS <- trainControl(method="repeatedcv", 
                              number=10, 
                              repeats=3, 
                              search="random",
                              verboseIter = TRUE)
    set.seed(111)

    rf_random <- train(classe~., 
                       data=train.set, 
                       method="rf", 
                       metric=metric, 
                       tuneLength=20, 
                       trControl=controlRS)

    save(rf_random, file = "rf_random.RData")

![](PaoloCoraggioFinalAssignment_files/figure-markdown_strict/unnamed-chunk-24-1.png)

![](PaoloCoraggioFinalAssignment_files/figure-markdown_strict/unnamed-chunk-25-1.png)

The latest 2 plots suggests that a `ntree` = 100 and `mtry` = 10 should
speed up the processing while assuring a better accuracy. The following
is the model with these parameters and its accuracy on the testing set.

    set.seed(111)
    mod.RFfinal <- train(classe ~., data = train.set,
                         method = "rf",
                         ntree = 100,
                         tuneGrid = data.frame(mtry=10),
                         trControl = trainControl(method = "repeatedcv", 
                                                  number=10,
                                                  repeats=3,
                                                  verboseIter = TRUE))

    confusionMatrix(predict(mod.RFfinal, validation.set), 
                validation.set$classe)$overall

<table>
<thead>
<tr class="header">
<th style="text-align: right;">Accuracy</th>
<th style="text-align: right;">Kappa</th>
<th style="text-align: right;">AccuracyLower</th>
<th style="text-align: right;">AccuracyUpper</th>
<th style="text-align: right;">AccuracyNull</th>
<th style="text-align: right;">AccuracyPValue</th>
<th style="text-align: right;">McnemarPValue</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.994</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">0.997</td>
<td style="text-align: right;">0.284</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">NaN</td>
</tr>
</tbody>
</table>

And further by class.

<table>
<thead>
<tr class="header">
<th></th>
<th style="text-align: right;">Sensitivity</th>
<th style="text-align: right;">Specificity</th>
<th style="text-align: right;">Pos Pred Value</th>
<th style="text-align: right;">Neg Pred Value</th>
<th style="text-align: right;">Precision</th>
<th style="text-align: right;">Recall</th>
<th style="text-align: right;">F1</th>
<th style="text-align: right;">Prevalence</th>
<th style="text-align: right;">Detection Rate</th>
<th style="text-align: right;">Detection Prevalence</th>
<th style="text-align: right;">Balanced Accuracy</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Class: A</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.999</td>
<td style="text-align: right;">0.998</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.998</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.999</td>
<td style="text-align: right;">0.284</td>
<td style="text-align: right;">0.284</td>
<td style="text-align: right;">0.285</td>
<td style="text-align: right;">1.000</td>
</tr>
<tr class="even">
<td>Class: B</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">0.999</td>
<td style="text-align: right;">0.997</td>
<td style="text-align: right;">0.998</td>
<td style="text-align: right;">0.997</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.194</td>
<td style="text-align: right;">0.192</td>
<td style="text-align: right;">0.193</td>
<td style="text-align: right;">0.996</td>
</tr>
<tr class="odd">
<td>Class: C</td>
<td style="text-align: right;">0.997</td>
<td style="text-align: right;">0.996</td>
<td style="text-align: right;">0.982</td>
<td style="text-align: right;">0.999</td>
<td style="text-align: right;">0.982</td>
<td style="text-align: right;">0.997</td>
<td style="text-align: right;">0.989</td>
<td style="text-align: right;">0.174</td>
<td style="text-align: right;">0.174</td>
<td style="text-align: right;">0.177</td>
<td style="text-align: right;">0.997</td>
</tr>
<tr class="even">
<td>Class: D</td>
<td style="text-align: right;">0.990</td>
<td style="text-align: right;">0.999</td>
<td style="text-align: right;">0.996</td>
<td style="text-align: right;">0.998</td>
<td style="text-align: right;">0.996</td>
<td style="text-align: right;">0.990</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">0.164</td>
<td style="text-align: right;">0.162</td>
<td style="text-align: right;">0.163</td>
<td style="text-align: right;">0.994</td>
</tr>
<tr class="odd">
<td>Class: E</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.998</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.993</td>
<td style="text-align: right;">0.996</td>
<td style="text-align: right;">0.184</td>
<td style="text-align: right;">0.182</td>
<td style="text-align: right;">0.182</td>
<td style="text-align: right;">0.996</td>
</tr>
</tbody>
</table>

As we can see from the table, the improvement is really minimal in terms
of accuracy although the algorithm speed is improved a lot.

The following plot shows also the top 15 features in terms of
importance.

![](PaoloCoraggioFinalAssignment_files/figure-markdown_strict/unnamed-chunk-29-1.png)

Applying the model to the test set
==================================

Finally, the model is applied to predict the data in the test set.

    predict(mod.RF, newdata = testset)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

References and Future Work
==========================

-   The data used are part of the [WLE
    dataset](http://groupware.les.inf.puc-rio.br/har)
-   For the Random Forest tuning with caret I read \[James Brownlee blog
    page\]
    (<a href="https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/" class="uri">https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/</a>)
-   A very helpful resource for this project has been [Applied
    Predictive Modeling](http://appliedpredictivemodeling.com/) by Max
    Kuhn and Kjell Johnson (both book and related blog)

I will keep the project updated on the GitHub repository as I will use
the proposed assignment to perform further analysis (e.g. PCA and
prediction based on temporal series).
