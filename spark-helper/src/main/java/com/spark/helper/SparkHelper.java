package com.spark.helper;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;

/***
 * Containd Helper methods for modules.
 */
public class SparkHelper {

    public static SparkSession getSparkSession(String appName, String master) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        return SparkSession.builder()
                .appName(appName)
                .master(master).getOrCreate();
    }
}
