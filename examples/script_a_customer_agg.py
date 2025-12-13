from pyspark.sql import functions as F

df1 = spark.read.parquet("s3a://mybucket/landing/customer_landing.parquet")

dfAgg = (df1
  .groupBy("customer_id","product_id","event_date")
  .agg(
    F.sum("amount").alias("total_amount"),
    F.sum("revenue").alias("total_revenue")
  )
  .withColumn("big_amount", F.col("total_amount") + F.col("total_revenue"))
)

dfCurated = dfAgg
dfCurated.write.mode("overwrite").parquet("s3a://mybucket/curated/customer_agg/")
