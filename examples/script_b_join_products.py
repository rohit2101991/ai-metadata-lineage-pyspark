from pyspark.sql import functions as F

df3 = spark.read.parquet("s3a://mybucket/curated/customer_agg/")
dfProducts = spark.read.parquet("s3a://mybucket/ref/products.parquet")

dfJoined = df3.join(dfProducts, on="product_id", how="left")

dfStage = (dfJoined
  .withColumn("margin", F.col("big_amount") - F.col("price"))
  .withColumn("is_profitable", F.when(F.col("margin") > 0, F.lit(True)).otherwise(F.lit(False)))
)

dfStage.write.mode("overwrite").parquet("s3a://mybucket/stage/cust_products/")
