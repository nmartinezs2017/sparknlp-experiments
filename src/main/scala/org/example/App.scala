package org.example

import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotators.{LemmatizerModel, Stemmer, StopWordsCleaner, Tokenizer}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, ByteType, DoubleType, IntegerType, StructField, StructType}

import scala.collection.mutable
import scala.io.Source

/**
 * Spark NLP Experiments
 *
 */
object App extends App {
  // Create context
  System.setProperty("hadoop.home.dir", "C:\\Users\\USER\\IdeaProjects\\spark-nlp")
  val spark = SparkSession.builder.appName("SparkNLP-Scala-Experiments")
    .master("local[*]")
    .getOrCreate()

  // Read Tweet Dataset
  val df = spark.read.option("header",true).csv("data/tweet_dataset.csv")

  df.show()

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
  val bufferedSource = Source.fromFile("spanish.txt")
  val stopwords = bufferedSource.getLines().toArray

  // StopWordsCleaner
  val stopWordsCleaner = new StopWordsCleaner()
    .setInputCols("token")
    .setOutputCol("cleanTokens")
    .setStopWords(stopwords)
    .setCaseSensitive(false)

  // Stemmer. Returns hard-stems out of words with the objective of retrieving the meaningful part of the word
  val stemmer = new Stemmer()
    .setInputCols(Array("cleanTokens"))
    .setOutputCol("stem")

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
    fit(df).
    transform(df).toDF()

  val tokens_df = preprocessed_df.select("tokens")

  //////// TOKENIZATION /////////

  val cv = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(100)
  val cv_model = cv.fit(tokens_df)
  // transform the data. Output column name will be features.
  val vectorized_tokens = cv_model.transform(tokens_df)
  println("Tokenization result")
  vectorized_tokens.show()

  //////// TRANING /////////
  val num_topics = 6
  val lda = new LDA().setK(num_topics).setMaxIter(10)
  val model = lda.fit(vectorized_tokens)
  val ll = model.logLikelihood(vectorized_tokens)
  val lp = model.logPerplexity(vectorized_tokens)
  println("The lower bound on the log likelihood of the entire corpus: " + ll)
  println("The upper bound on perplexity: " + lp)

  //////// SHOW RESULTS /////////
  val vocab = cv_model.vocabulary
  val topics = model.describeTopics()
  val topics_rdd = topics.rdd
  topics.show()

  val topics_words = topics_rdd
    .map(row => row.getAs[mutable.WrappedArray[Int]]("termIndices"))
    .map(idx_list => idx_list.map(idx => vocab.lift(idx)))

  topics_words.foreach(array_token => println(array_token))

}
