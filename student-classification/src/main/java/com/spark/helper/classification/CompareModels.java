package com.spark.helper.classification;

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * Class that compares different models - Decision tree and Random Forest.
 * Here, the dataset is read, pipeline is created and the obtained predictions
 * are evaluated based on their f-score.
 */
public class CompareModels {
    public static final String DATASET_PATH = "student-classification\\src\\main\\resources\\UserKnowModelingDataset_Train.csv";

    public static void main(String[] args) {
        /***Read the dataset***/
        StudentClassification classification = new StudentClassification();
        Dataset<Row> dataset = classification.readDataset(DATASET_PATH);

        /***Dataset read is split into training and test data randomly.***/
        Dataset<Row>[] datasets = dataset.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = datasets[0];
        Dataset<Row> testData = datasets[1];

        /***Create a pipeline for processing the dataset***/
        Dataset<Row> predictionsDecisionTree = classification.createAndExecutePipeline(ClassifierAlgorithm.DECISIONTREE, trainingData, testData);
        Dataset<Row> predictionsRandomForest = classification.createAndExecutePipeline(ClassifierAlgorithm.RANDAOMFOREST, trainingData, testData);

        /***Fetch the label and prediction columns***/
        LabelAndPredictionColumn labelAndPredictionColumn = classification.getLabelAndPredictionColumn();

        /**Evaluate the dataset based on their f-score**/
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol(labelAndPredictionColumn.getLabelColumnName())
                .setPredictionCol(labelAndPredictionColumn.getPredictionColumnName());
        double decisionTreeAccuracy = evaluator.evaluate(predictionsDecisionTree);
        double randomForestAccuracy = evaluator.evaluate(predictionsRandomForest);

        System.out.println("f-score for Random Forest = " + (randomForestAccuracy));
        System.out.println("f-score for Decision Tree = " + (decisionTreeAccuracy));

    }
}
