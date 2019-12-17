package com.spark.common;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/***
 * Containd Helper methods for modules.
 */
public class HelperMethods {

    public static SparkSession getSparkSession(String appName, String master) {
        /***Setting default spark log level to error***/
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        /***Creates and returns the Spark session object***/
        return SparkSession.builder()
                .appName(appName)
                .master(master).getOrCreate();
    }

    public static boolean isNullOrEmpty(String str) {
        return str == null || str.isEmpty() || str.trim().length() == 0;
    }

    public static Dataset<Row> readDataset(String appName, String master, String datasetPath) {
        /***Create the Spark session***/
        SparkSession sparkSession = HelperMethods.
                getSparkSession(appName, master);

        /***Read the dataset***/
        Dataset<Row> dataset = sparkSession.read().option("header", true).option("inferschema", true).csv(datasetPath);

        /***Display first 20 rows, the schema and basic stats for the dataset***/
        dataset.show();
        dataset.printSchema();
        dataset.describe().show();
        return dataset;
    }
}
