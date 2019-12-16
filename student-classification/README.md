The dataset for this project is present in resources section. It consists of the following columns:
1. SST - time spent studying a subject
2. SRT - time spent on revision before exam
3. SAT - assignment score
4. SAP - assignment performance
5. SEP - exam performance
6. SKL - knowledge level
Problem: Classify students based on their knowledge level into very_low, low, medium or high categories.

The project makes use of Spark-ML library to implement and compare 2 models - Decision Tree and Random Forest.
The F-score is used as an evaluation measure to compare the results from both the predictions.