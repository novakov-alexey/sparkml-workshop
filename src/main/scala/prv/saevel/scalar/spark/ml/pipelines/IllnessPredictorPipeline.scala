package prv.saevel.scalar.spark.ml.pipelines

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature._

object IllnessPredictorPipeline {
  /**
    * Constructs an ML Pipeline that will transform the data into a Vector form names "features" defined as:
    * features(0) = sex field indexed
    * features(1) = weigth
    * features(2) = height
    * features(3) = otherIndicators + otherIndicators^2 + otherIndicators^3
    * features(4) = age + age^2 + age^3
    * and use a <code>DecisionTreeClassifier</code> to predict whether the patient is ill and puts them in the
    * "illness_prediction" column.
    */
  def apply(): Pipeline = {
    new Pipeline().setStages(Array(
      new StringIndexer().setInputCol("sex").setOutputCol("sex_indexed"),
      new VectorAssembler().setInputCols(Array("age", "otherIndicators")).setOutputCol("mini_vector"),
      new PolynomialExpansion().setInputCol("mini_vector").setOutputCol("poly_mini_vector").setDegree(3),
      new VectorAssembler().setInputCols(Array("sex_indexed", "weigth", "height", "poly_mini_vector")).setOutputCol("features"),
      new DecisionTreeClassifier().setLabelCol("illness_actual")
        .setFeaturesCol("features")
        .setPredictionCol("illness_prediction")
    ))
  }
}