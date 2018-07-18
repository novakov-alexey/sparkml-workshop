package prv.saevel.scalar.spark.ml.crossvalidation

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object CrossValidatorPipeline {
  /**
    * Returns a <code>CrossValidator</code> with a <code>Pipeline</code> inside, which takes <code>IncomeData</code> elements
    * and build the "features" Vector as follows:
    *   features(0) = indexed "sex" field
    *   features(1) = indexed "educationLevel" field
    *   features(2) = indexed "fieldOfExpertise" field
    *   features(3) = age
    *   features(4) = yearsOfExperience
    *
    *   and then processes passes it through a <code>RandomForestClassifier</code>, with label column "incomeBracket"
    *   and prediction column "predicted_income_bracket".
    *
    *   The <code>CrossValidator</code> trains / validates the "numTrees" values for the <code>RandomForestClassifier</code>
    *   from the values in <code>possibleNumTress</code>.
    */
  def apply(possibleNumTrees: Array[Int]): CrossValidator = {
    val classifier = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("incomeBracket")
      .setPredictionCol("predicted_income_bracket")

    val evaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol("predicted_income_bracket")
      .setLabelCol("incomeBracket")
      .setMetricName("accuracy")

    val pipeline = new Pipeline().setStages(Array(
      new StringIndexer()
        .setInputCol("sex")
        .setOutputCol("sex_indexed"),
      new StringIndexer()
        .setInputCol("educationLevel")
        .setOutputCol("edu_level_indexed"),
      new StringIndexer()
        .setInputCol("fieldOfExpertise")
        .setOutputCol("field_indexed"),
      new VectorAssembler()
        .setInputCols(Array("sex_indexed", "edu_level_indexed", "field_indexed", "age", "yearsOfExperience"))
        .setOutputCol("features"),
      classifier
    ))

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.numTrees, possibleNumTrees)
      .build()

    new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
  }
}