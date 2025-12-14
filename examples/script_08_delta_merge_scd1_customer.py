from pyspark.sql import functions as F
from delta.tables import DeltaTable

def run(spark):
    updates = spark.read.parquet("s3a://my-bucket/silver/dim_customer/")  # latest snapshot
    target_path = "s3a://my-bucket/delta/dim_customer_scd1/"

    # Ensure target exists (first run bootstrap)
    if not DeltaTable.isDeltaTable(spark, target_path):
        (updates
         .withColumn("ingested_at", F.current_timestamp())
         .write.format("delta").mode("overwrite").save(target_path))

    tgt = DeltaTable.forPath(spark, target_path)

    (tgt.alias("t")
        .merge(
            updates.alias("s"),
            "t.customer_id = s.customer_id"
        )
        .whenMatchedUpdate(set={
            "email": "s.email",
            "customer_tier": "s.customer_tier",
            "lifetime_value": "s.lifetime_value",
            "updated_at": "s.updated_at",
            "ingested_at": "current_timestamp()"
        })
        .whenNotMatchedInsert(values={
            "customer_id": "s.customer_id",
            "email": "s.email",
            "customer_tier": "s.customer_tier",
            "lifetime_value": "s.lifetime_value",
            "updated_at": "s.updated_at",
            "ingested_at": "current_timestamp()"
        })
        .execute())

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .appName("delta_merge_scd1_customer")
             .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
             .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
             .getOrCreate())
    run(spark)


