package prv.saevel.scalar.spark.ml.utils

import org.apache.spark.ml.feature.{Binarizer, Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset}
import prv.saevel.scalar.spark.ml.Patient

object PatientPreprocessor {
  /**
    * Preprocesses a <code>Dataset[Patient]</code>, by adding a "features" (Vector) column, constructed as follows:
    *
    * features(0) = height
    * features(1) = weigth
    * features(2) = sex field, indexed into Doubles
    * features(3) = age field, bucketized into the following buckets: [0, 10], [10, 20], [20, 40], [40, 70], [70, 100]
    * features(4) = 1.0 if "otherIndicators" > 0.75, 0.0 otherwise.
    */
  def preprocess(patients: Dataset[Patient]): DataFrame = {
    val indexer = new StringIndexer().setInputCol("sex").setOutputCol("sex_indexed")
    val bucketizer = new Bucketizer().setInputCol("age").setSplits(Array(0.0, 10.0, 20.0, 40.0, 70.0, 100.0)).setOutputCol("age_bucketized")
    val binarizer = new Binarizer().setInputCol("otherIndicators").setThreshold(0.75).setOutputCol("other")
    val assembler = new VectorAssembler().setInputCols(Array("height", "weigth", "sex_indexed", "age_bucketized", "other")).setOutputCol("features")

    val df1 = indexer.fit(patients).transform(patients)
    val df2 = bucketizer.transform(df1)
    val df3 = binarizer.transform(df2)

    assembler.transform(df3)
  }
}