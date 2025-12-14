from pyspark.sql import functions as F, Window

def run(spark):
    fact = spark.read.parquet("s3a://my-bucket/gold/fact_orders_enriched/")
    sessions = spark.read.parquet("s3a://my-bucket/gold/web_sessions/")

    # Join sessions to orders by customer + date as a proxy
    df = (fact.alias("f")
          .join(sessions.alias("s"),
                (F.col("f.customer_id") == F.col("s.customer_id")), "left")
          .withColumn("high_value_flag", F.when(F.col("f.abs_amount") >= 500, 1).otherwise(0))
          .withColumn("risk_from_channel",
                      F.when(F.col("f.channel").isin("unknown", "affiliate"), 2)
                       .when(F.col("f.channel") == "web", 1)
                       .otherwise(0))
          .withColumn("risk_from_refund", F.when(F.col("f.is_refund") == 1, 3).otherwise(0))
          .withColumn("risk_from_sessions",
                      F.when(F.col("s.events") >= 50, 2)
                       .when(F.col("s.events") >= 20, 1)
                       .otherwise(0))
          .withColumn("raw_risk_score",
                      F.col("high_value_flag") + F.col("risk_from_channel") + F.col("risk_from_refund") + F.col("risk_from_sessions"))
    )

    w = Window.partitionBy("order_date").orderBy(F.col("raw_risk_score").desc(), F.col("abs_amount").desc())

    df2 = (df
        .withColumn("risk_rank_day", F.rank().over(w))
        .withColumn("risk_percent_rank_day", F.percent_rank().over(w))
        .withColumn("risk_bucket",
                    F.when(F.col("risk_percent_rank_day") >= 0.99, "critical")
                     .when(F.col("risk_percent_rank_day") >= 0.95, "high")
                     .when(F.col("risk_percent_rank_day") >= 0.80, "medium")
                     .otherwise("low"))
    )

    out = "s3a://my-bucket/gold/fact_orders_risk_scored/"
    df2.write.mode("overwrite").partitionBy("order_date").parquet(out)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("fraud_scoring").getOrCreate()
    run(spark)



