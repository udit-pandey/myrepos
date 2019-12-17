package com.spark.spamclassifier;

import com.spark.common.ClassifierAlgorithm;
import com.spark.common.HelperMethods;
import com.spark.common.LabelAndPredictionColumn;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class SpamClassifier {
    private LabelAndPredictionColumn labelAndPredictionColumn = new LabelAndPredictionColumn();

    public Dataset<Row> readDatasetAndPrintStats(String appName, String master, String datasetPath) {
        /***Read the dataset and print basic stats for it***/
        Dataset<Row> dataset = HelperMethods.readDataset(appName, master, datasetPath);
        dataset.groupBy("v1").count().show();
        return dataset;
    }

    public Dataset<Row> createAndExecutePipeline(ClassifierAlgorithm algorithm, Dataset<Row> dataset) {
        /***Dropping rows with null values in column v1 and v2***/
        dataset = dataset.na().drop(new String[]{"v1", "v2"});

        /***Dataset is split into training and test data randomly.***/
        Dataset<Row>[] randomSplit = dataset.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = randomSplit[0];
        Dataset<Row> testData = randomSplit[1];

        /***Convert String labels(SPAM, HAM in this case) to double values***/
        StringIndexerModel stringIndexerModel = new StringIndexer()
                .setInputCol("v1")
                .setOutputCol("label")
                .fit(trainingData);

        /***Extract words from message (Tokenize the input text)***/
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("v2")
                .setOutputCol("words");

        /***Remove the stop words (a commonly used word such as “the”, “a”, “an”, “in”)***/
        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("filtered");

        /***Create the Term Frequency Matrix***/
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(stopWordsRemover.getOutputCol())
                .setOutputCol("frequency");

        /***Calculate the Inverse Document Frequency***/
        IDF inverseDocumentFrequency = new IDF()
                .setInputCol(hashingTF.getOutputCol())
                .setOutputCol("features");

        /***Converting indexed labels back to original labels***/
        IndexToString indexToString = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(stringIndexerModel.labels());

        /***Model Building***/
        /***Creating a pipeline of operations based on the algorithm to be used***/
        Pipeline pipeline = new Pipeline();
        if (algorithm.equals(ClassifierAlgorithm.DECISIONTREE)) {
            DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier()
                    .setFeaturesCol(inverseDocumentFrequency.getOutputCol())
                    .setLabelCol(stringIndexerModel.getOutputCol());
            pipeline.setStages(new PipelineStage[]{stringIndexerModel, tokenizer,
                    stopWordsRemover, hashingTF, inverseDocumentFrequency, decisionTreeClassifier,
                    indexToString});
        } else if (algorithm.equals(ClassifierAlgorithm.RANDOMFOREST)) {
            RandomForestClassifier randomForestClassifier = new RandomForestClassifier()
                    .setFeaturesCol(inverseDocumentFrequency.getOutputCol())
                    .setLabelCol(stringIndexerModel.getOutputCol());
            pipeline.setStages(new PipelineStage[]{stringIndexerModel, tokenizer,
                    stopWordsRemover, hashingTF, inverseDocumentFrequency, randomForestClassifier,
                    indexToString});
        }

        /***Setting the label and prediction columns to use it for model evaluation***/
        labelAndPredictionColumn.setLabelColumnName(stringIndexerModel.getOutputCol());
        labelAndPredictionColumn.setPredictionColumnName(indexToString.getInputCol());

        /***Executing the pipeline operations***/
        Dataset<Row> predictions = pipeline.fit(trainingData).transform(testData);
        predictions.show();

        /***Printing the Confusion matrix based on the predictions obtained***/
        predictions.groupBy(stringIndexerModel.getInputCol(), indexToString.getOutputCol()).count().show();
        return predictions;
    }

    public LabelAndPredictionColumn getLabelAndPredictionColumn() {
        return labelAndPredictionColumn;
    }
}
