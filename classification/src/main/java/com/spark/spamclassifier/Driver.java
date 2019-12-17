package com.spark.spamclassifier;

import com.spark.common.ClassifierAlgorithm;
import com.spark.common.HelperMethods;
import com.spark.common.LabelAndPredictionColumn;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Driver {
    private static final String DATASET_PATH = "classification\\src\\main\\resources\\spam.csv";

    public static void main(String[] args) {
        /***Read the dataset***/
        SpamClassifier spamClassifier = new SpamClassifier();
        Dataset<Row> dataset = HelperMethods.readDataset("spamClassifier", "local[*]", DATASET_PATH);

        /***Create a pipeline for processing the dataset***/
        Dataset<Row> predictionRF = spamClassifier.createAndExecutePipeline(ClassifierAlgorithm.RANDOMFOREST, dataset);

        /***Fetch the label and prediction columns***/
        LabelAndPredictionColumn labelAndPredictionColumn = spamClassifier.getLabelAndPredictionColumn();

        /**Evaluate the model**/
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol(labelAndPredictionColumn.getLabelColumnName())
                .setPredictionCol(labelAndPredictionColumn.getPredictionColumnName());
        double fscore = evaluator.evaluate(predictionRF);
        System.out.println("F-Score for the model: " + fscore);
    }
}
