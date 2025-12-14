from pyspark.sql import functions as F, Window

def run(spark):
    orders = spark.read.parquet("s3a://my-bucket/bronze/orders/")
    dim_cust = spark.read.parquet("s3a://my-bucket/silver/dim_customer/")

    df = (orders.alias("o")
          .join(dim_cust.alias("c"), F.col("o.customer_id") == F.col("c.customer_id"), "left")
          .select(
              F.col("o.order_id"),
              F.col("o.customer_id"),
              F.col("c.customer_tier"),
              F.col("o.order_ts"),
              F.col("o.order_date"),
              F.col("o.country"),
              F.col("o.channel"),
              F.col("o.amount"),
              F.col("o.abs_amount"),
              F.col("o.is_refund")
          ))

    w = Window.partitionBy("customer_id").orderBy(F.col("order_ts").asc())

    df2 = (df
        .withColumn("prev_amount", F.lag("abs_amount", 1).over(w))
        .withColumn("next_amount", F.lead("abs_amount", 1).over(w))
        .withColumn("amount_delta_prev", F.col("abs_amount") - F.coalesce(F.col("prev_amount"), F.lit(0.0)))
        .withColumn("rolling_7_orders_sum",
                    F.sum("abs_amount").over(w.rowsBetween(-6, 0)))
        .withColumn("rolling_30_orders_sum",
                    F.sum("abs_amount").over(w.rowsBetween(-29, 0)))
    )

    out = "s3a://my-bucket/gold/fact_orders_enriched/"
    df2.write.mode("overwrite").partitionBy("order_date").parquet(out)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("fact_orders_enriched").getOrCreate()
    run(spark)


