
from pyspark.sql import functions as F, Window

def run(spark):
    src = "s3a://my-bucket/raw/customers/"
    df = spark.read.parquet(src)

    w = Window.partitionBy("customer_id").orderBy(F.col("updated_at").desc(), F.col("version").desc())

    df2 = (df
        .withColumn("rn", F.row_number().over(w))
        .withColumn("rnk", F.rank().over(w))
        .withColumn("drnk", F.dense_rank().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .withColumn("customer_tier",
                    F.when(F.col("lifetime_value") >= 10000, "platinum")
                     .when(F.col("lifetime_value") >= 5000, "gold")
                     .when(F.col("lifetime_value") >= 1000, "silver")
                     .otherwise("bronze"))
        .withColumn("email_domain", F.lower(F.regexp_extract(F.col("email"), r"@(.+)$", 1)))
    )

    out = "s3a://my-bucket/silver/dim_customer/"
    df2.write.mode("overwrite").parquet(out)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("dim_customer_latest").getOrCreate()
    run(spark)


