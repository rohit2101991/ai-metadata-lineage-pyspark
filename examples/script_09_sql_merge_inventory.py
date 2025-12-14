from pyspark.sql import functions as F

def run(spark):
    inv = spark.read.parquet("s3a://my-bucket/raw/inventory_updates/")  # sku, warehouse, on_hand, updated_at
    inv = inv.withColumn("updated_at", F.to_timestamp("updated_at"))

    inv.createOrReplaceTempView("inv_updates")

    spark.sql("""
      CREATE TABLE IF NOT EXISTS delta_inventory
      USING delta
      LOCATION 's3a://my-bucket/delta/inventory/'
    """)

    spark.sql("""
      MERGE INTO delta_inventory t
      USING inv_updates s
      ON t.sku = s.sku AND t.warehouse = s.warehouse
      WHEN MATCHED AND s.updated_at >= t.updated_at THEN
        UPDATE SET
          t.on_hand = s.on_hand,
          t.updated_at = s.updated_at
      WHEN NOT MATCHED THEN
        INSERT (sku, warehouse, on_hand, updated_at)
        VALUES (s.sku, s.warehouse, s.on_hand, s.updated_at)
    """)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .appName("sql_merge_inventory")
             .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
             .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
             .enableHiveSupport()
             .getOrCreate())
    run(spark)


