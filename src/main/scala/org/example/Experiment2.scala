package org.example

import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotators.{LemmatizerModel, StopWordsCleaner, Tokenizer}
import com.typesafe.config.ConfigFactory
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SparkSession

import java.io.File
import scala.io.Source
import scala.collection.mutable.ListBuffer

/**
 * Spark NLP Experiments
 *
 */
object Experiment2 extends App {

  // Import config
  val config = ConfigFactory.parseFile(new File("application.conf"))
  val sparkConfig = config.getConfig("conf.spark")
  val experiment2Config = config.getConfig("conf.experiment2")

  // Create context
  System.setProperty("hadoop.home.dir", sparkConfig.getString("HADOOP_DIR"))
  val spark = SparkSession.builder.appName("SparkNLP-Scala-Experiments")
    .master(sparkConfig.getString("master"))
    .getOrCreate()

  // Read Tweet Dataset
  val df_all = spark.read.option("header",true).csv(experiment2Config.getString("DATASET_PATH"))
  df_all.show()


  //////////// EXPERIMENT PREPARATION //////////////

  import org.apache.spark.sql.functions._
  import spark.implicits._

  val p = "^\\[(.*)\\]$"
  val df_aux = df_all
    .withColumn("other_hashtags", split(regexp_extract(col("other_hashtags"), p, 1), "\\s*,\\s*"))
    .select($"other_hashtags", $"preprocessed", explode($"other_hashtags"))
  val df_explode = df_aux.filter(length(df_aux("col")) >= 1)

  df_explode.show()

  val df_order_hashtags = df_explode.groupBy("col").count().sort(desc("count"))
  df_order_hashtags.show()

  val total = df_order_hashtags.select(sum("count")).first().getLong(0)
  val df_percentage = df_order_hashtags.withColumn("count", $"count" / total)
  df_percentage.show()

  val df_top_hashtags = df_percentage.filter(df_percentage("count") > 0.01)
  df_top_hashtags.show(false)

  val numTopics = df_top_hashtags.count()

  println(s"Num topics: $numTopics")

  // Filtrar tuits
  val list_top=df_top_hashtags.select("col").map(f=>f.getString(0))
    .collect.toList

  val df_only_top = df_explode.filter(df_explode("col").isin(list_top : _*)).distinct()

  df_only_top.show()

  //////// PREPROCESSING /////////

  // DocumentAssembler
  val documentAssembler = new DocumentAssembler()
    .setInputCol("preprocessed")
    .setOutputCol("document")
    .setCleanupMode("shrink")

  // Tokenizer. Identifies tokens with tokenization open standards.
  val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  // Stop words dictionary
  val bufferedSource = Source.fromFile("stopwordsdict.txt")
  val stopwords = bufferedSource.getLines().toArray

  // StopWordsCleaner
  val stopWordsCleaner = new StopWordsCleaner()
    .setInputCols("token")
    .setOutputCol("cleanTokens")
    .setStopWords(stopwords)
    .setCaseSensitive(false)

  // Lemmatizer. Retrieves lemmas out of words with the objective of returning a base dictionary word
  val lemmatizer = LemmatizerModel.pretrained("lemma", "es")
    .setInputCols(Array("cleanTokens"))
    .setOutputCol("lemma")

  // Finisher. Helps to clean the metadata and output the results into an array
  val finisher = new Finisher()
    .setInputCols(Array("lemma"))
    .setOutputCols(Array("tokens"))
    .setOutputAsArray(true)
    .setCleanAnnotations(false)

  // Pipeline preprocessing
  val pipeline = new Pipeline().
    setStages(Array(
      documentAssembler,
      tokenizer,
      stopWordsCleaner,
      lemmatizer,
      finisher
    ))

  val preprocessed_df = pipeline.
    fit(df_all).
    transform(df_all).toDF()

  preprocessed_df.show()
  preprocessed_df.select("cleanTokens").show(20, false)
  preprocessed_df.select("lemma").show(20, false)
  val tokens_df = preprocessed_df.select("tokens")

  //////// TOKENS -> FEATURES /////////

  val cv = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(experiment2Config.getInt("minDF"))
  val cv_model = cv.fit(tokens_df)
  // transform the data. Output column name will be features.
  val vectorized_tokens = cv_model.transform(tokens_df)
  println("Tokenization result")
  vectorized_tokens.show()

  //////// TRANING /////////
  val num_topics = experiment2Config.getInt("numTopics")
  val lda = new LDA().setK(num_topics).setMaxIter(experiment2Config.getInt("maxIter"))
  val model = lda.fit(vectorized_tokens)

  //////// SHOW RESULTS /////////
  val ll = model.logLikelihood(vectorized_tokens)
  val lp = model.logPerplexity(vectorized_tokens)
  println("The lower bound on the log likelihood of the entire corpus: " + ll)
  println("The upper bound on perplexity: " + lp)

  val vocab = cv_model.vocabulary
  val topics = model.describeTopics(experiment2Config.getInt("maxTermsPerTopic"))
  val topics_rdd = topics.rdd
  topics.show()
  topics

  val topics_words = topics_rdd
    .map(row => row.getAs[scala.collection.mutable.WrappedArray[Int]]("termIndices"))
    .map(idx_list => idx_list.map(idx => vocab.lift(idx)))

  var results_text: ListBuffer[String] = ListBuffer()

  println("|------ TOPICS ------|")

  topics_words.foreach(array_token => {
    val array_words = array_token.map({ case Some(word) => word })
    val output_topic = array_words.mkString(" ")
    println(output_topic)
    results_text.insert(0, output_topic) // The topic is inserted in the list of results.
  }
  )

  //////// SAVE RESULTS /////////
  import java.io._
  val pw = new PrintWriter(new File(experiment2Config.getString("RESULTS_PATH")))
  pw.write(results_text.mkString("\n"))
  pw.close

}
