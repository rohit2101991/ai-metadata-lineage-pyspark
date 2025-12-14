from pyspark.sql import functions as F, Window

def run(spark):
    fact = spark.read.parquet("s3a://my-bucket/gold/fact_orders_enriched/")
    dim_prod = spark.read.parquet("s3a://my-bucket/silver/dim_product/")
    line_items = spark.read.parquet("s3a://my-bucket/raw/order_items/")  # order_id, product_id, qty

    df = (fact.alias("f")
          .join(line_items.alias("i"), "order_id", "inner")
          .join(dim_prod.alias("p"), "product_id", "left")
          .withColumn("gross_sales", F.col("f.abs_amount"))
          .withColumn("qty", F.coalesce(F.col("qty").cast("int"), F.lit(1)))
          .withColumn("est_cost", F.col("qty") * F.coalesce(F.col("p.cost"), F.lit(0.0)))
          .withColumn("est_margin", F.col("gross_sales") - F.col("est_cost"))
          .withColumn("est_margin_pct",
                      F.when(F.col("gross_sales") == 0, F.lit(None))
                       .otherwise(F.col("est_margin") / F.col("gross_sales")))
    )

    w = Window.partitionBy("order_date").orderBy(F.col("est_margin").desc())

    df2 = (df
        .withColumn("margin_rank_day", F.rank().over(w))
        .withColumn("margin_dense_rank_day", F.dense_rank().over(w))
        .withColumn("top_10_flag", F.when(F.col("margin_rank_day") <= 10, 1).otherwise(0))
    )

    out = "s3a://my-bucket/gold/sales_margin_ranked/"
    df2.write.mode("overwrite").partitionBy("order_date").parquet(out)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("sales_margin_rank").getOrCreate()
    run(spark)



