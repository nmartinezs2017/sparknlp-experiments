package org.example

import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotators.{LemmatizerModel, StopWordsCleaner, Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SparkSession
import scala.io.Source
import scala.collection.mutable.ListBuffer

/**
 * Spark NLP Experiments
 *
 */
object Experiment1 extends App {
  // Create context
  System.setProperty("hadoop.home.dir", "C:\\Users\\USER\\IdeaProjects\\spark-nlp")
  val spark = SparkSession.builder.appName("SparkNLP-Scala-Experiments")
    .master("local[*]")
    .getOrCreate()

  // Read Tweet Dataset
  val df = spark.read.option("header",true).csv("data/experimento1/dataset_5000f_14h.csv")
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
    fit(df).
    transform(df).toDF()

  preprocessed_df.show()
  preprocessed_df.select("cleanTokens").show(20, false)
  preprocessed_df.select("lemma").show(20, false)
  val tokens_df = preprocessed_df.select("tokens")

  //////// TOKENS -> FEATURES /////////

  val cv = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(50)
  val cv_model = cv.fit(tokens_df)
  // transform the data. Output column name will be features.
  val vectorized_tokens = cv_model.transform(tokens_df)
  println("Tokenization result")
  vectorized_tokens.show()

  //////// TRANING /////////
  val num_topics = 14
  val lda = new LDA().setK(num_topics).setMaxIter(100) // 500 antes
  val model = lda.fit(vectorized_tokens)

  //////// SHOW RESULTS /////////
  val ll = model.logLikelihood(vectorized_tokens)
  val lp = model.logPerplexity(vectorized_tokens)
  println("The lower bound on the log likelihood of the entire corpus: " + ll)
  println("The upper bound on perplexity: " + lp)

  val vocab = cv_model.vocabulary
  val topics = model.describeTopics(20)
  val topics_rdd = topics.rdd
  topics.show()

  val topics_words = topics_rdd
    .map(row => row.getAs[scala.collection.mutable.WrappedArray[Int]]("termIndices"))
    .map(idx_list => idx_list.map(idx => vocab.lift(idx)))

  var results_text: ListBuffer[String] = ListBuffer()

  topics_words.foreach(array_token => {
    val array_words = array_token.map({ case Some(word) => word })
    val output_topic = array_words.mkString(" ")
    println(output_topic)
    results_text.insert(0, output_topic) // The topic is inserted in the list of results.
  }
  )

  //////// SAVE RESULTS /////////
  import java.io._
  val pw = new PrintWriter(new File("20000f_14h.txt"))
  pw.write(results_text.mkString("\n"))
  pw.close

  /**
  topics_words.zipWithIndex.foreach(anyRef => {
    val array_token = anyRef._1
    val array_words = array_token.map({ case Some(word) => word })
    val output_topic = anyRef._2 + "\n" + array_words.mkString(" ")
    println(output_topic)
  }
  )**/


}
