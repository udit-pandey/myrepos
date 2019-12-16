package com.spark.helper.classification;

public class LabelAndPredictionColumn {
    private String labelColumnName;
    private String predictionColumnName;

    public String getLabelColumnName() {
        return labelColumnName;
    }

    public void setLabelColumnName(String labelColumnName) {
        this.labelColumnName = labelColumnName;
    }

    public String getPredictionColumnName() {
        return predictionColumnName;
    }

    public void setPredictionColumnName(String predictionColumnName) {
        this.predictionColumnName = predictionColumnName;
    }
}
