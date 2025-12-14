from pyspark.sql import functions as F

def run(spark):
    src = "s3a://my-bucket/raw/orders/"

    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(src))

    df1 = (df
        .withColumn("order_ts", F.to_timestamp("order_ts"))
        .withColumn("order_date", F.to_date("order_ts"))
        .withColumn("customer_id", F.col("customer_id").cast("string"))
        .withColumn("order_id", F.col("order_id").cast("string"))
        .withColumn("amount", F.coalesce(F.col("amount").cast("double"), F.lit(0.0)))
        .withColumn("currency", F.upper(F.coalesce(F.col("currency"), F.lit("USD"))))
        .withColumn("channel", F.lower(F.coalesce(F.col("channel"), F.lit("unknown"))))
        .withColumn("country", F.upper(F.coalesce(F.col("country"), F.lit("US"))))
        .withColumn("is_refund", F.when(F.col("amount") < 0, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("abs_amount", F.abs(F.col("amount")))
    )

    out = "s3a://my-bucket/bronze/orders/"
    (df1.write.mode("overwrite")
        .partitionBy("order_date", "country")
        .parquet(out))

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("bronze_ingest_orders").getOrCreate()
    run(spark)


