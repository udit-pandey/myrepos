package com.spark.helper.classification;

import com.spark.helper.SparkHelper;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

/***
 * Contains methods to read the dataset and analyse it.
 */
public class StudentClassification {
    private LabelAndPredictionColumn labelAndPredictionColumn = new LabelAndPredictionColumn();

    public Dataset<Row> readDataset(String datasetPath) {
        /***Create the Spark session***/
        SparkSession sparkSession = SparkHelper.
                getSparkSession("StudentClassification", "local[*]");

        /***Read the dataset***/
        Dataset<Row> dataset = sparkSession.read().option("header", true).option("inferschema", true).csv(datasetPath);
        
        /***Display first 20 rows, the schema and basic stats for the dataset***/
        dataset.show();
        dataset.printSchema();
        dataset.describe().show();
        dataset.groupBy("SKL").count().show();
        return dataset;
    }

    public Dataset<Row> createAndExecutePipeline(ClassifierAlgorithm classifierAlgorithm, Dataset<Row> trainingData, Dataset<Row> testData) {
        /***Convert String labels(the categorical data, student knowledge level in this case) to double values***/
        StringIndexerModel stringIndexerModel = new StringIndexer()
                .setInputCol("SKL")
                .setOutputCol("IND_SKL")
                .fit(trainingData);

        /***Converting required features into a dense vector***/
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"SST", "SRT", "SAT", "SAP", "SEP"})
                .setOutputCol("features");

        /***Converting indexed labels back to original labels***/
        IndexToString indexToString = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("LABEL_SKL")
                .setLabels(stringIndexerModel.labels());

        /***Model Building***/
        /***Creating a pipeline of operations based on the algorithm to be used***/
        Pipeline pipeline = new Pipeline();
        if (classifierAlgorithm.equals(ClassifierAlgorithm.DECISIONTREE)) {
            DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier()
                    .setLabelCol(stringIndexerModel.getOutputCol())
                    .setFeaturesCol(vectorAssembler.getOutputCol());
            pipeline.setStages(
                    new PipelineStage[]{stringIndexerModel, vectorAssembler, decisionTreeClassifier, indexToString});
        } else if (classifierAlgorithm.equals(ClassifierAlgorithm.RANDAOMFOREST)) {
            RandomForestClassifier randomForestClassifier = new RandomForestClassifier()
                    .setLabelCol(stringIndexerModel.getOutputCol())
                    .setFeaturesCol(vectorAssembler.getOutputCol());
            pipeline.setStages(
                    new PipelineStage[]{stringIndexerModel, vectorAssembler, randomForestClassifier, indexToString});
        }
        
        /***Executing the pipeline operations***/
        Dataset<Row> transformedDataset = pipeline.fit(trainingData).transform(testData);
        transformedDataset.show();

        /***Printing the Confusion matrix based on the predictions obtained***/
        System.out.println("Confusion Matrix:");
        transformedDataset.groupBy(col("SKL"), col("LABEL_SKL")).count().show();

        /***Setting the label and prediction columns to use it for model evaluation***/
        labelAndPredictionColumn.setLabelColumnName(stringIndexerModel.getOutputCol());
        labelAndPredictionColumn.setPredictionColumnName(indexToString.getInputCol());

        return transformedDataset;
    }

    public LabelAndPredictionColumn getLabelAndPredictionColumn() {
        return labelAndPredictionColumn;
    }
}
