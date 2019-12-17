This module solves 2 problems:

Problem 1: Classify students based on their knowledge level into very_low, low, medium or high categories.
The dataset for this problem consists of the following columns:
1. SST - time spent studying a subject
2. SRT - time spent on revision before exam
3. SAT - assignment score
4. SAP - assignment performance
5. SEP - exam performance
6. SKL - knowledge level
The project makes use of Spark-ML library to implement and compare 2 models - Decision Tree and Random Forest.
The F-score is used as an evaluation measure to compare the results from both the predictions.

Problem 2: Implement a spam classifier for mail.
The dataset for this problem consists of the following columns:
1. v1 - the category - SPAM or HAM.
2. v2 - message body
The project makes use of Spark-ML library to implement Random Forest classifier model which is more accurate
than Decision Forest classifier model.
Also, the datasets for both these sub-modules are present in the resources section of this module.