from pyspark.sql import functions as F

def run(spark):
    src = "s3a://my-bucket/raw/products/"
    df = spark.read.parquet(src)

    df.createOrReplaceTempView("raw_products")

    sql = """
    WITH cleaned AS (
      SELECT
        CAST(product_id AS STRING) AS product_id,
        LOWER(TRIM(category)) AS category,
        CAST(price AS DOUBLE) AS price,
        CAST(cost AS DOUBLE) AS cost,
        UPPER(COALESCE(currency, 'USD')) AS currency,
        TO_DATE(updated_at) AS updated_date,
        CASE
          WHEN price IS NULL OR price <= 0 THEN 0
          ELSE price
        END AS price_norm
      FROM raw_products
    ),
    metrics AS (
      SELECT
        product_id,
        category,
        currency,
        updated_date,
        price_norm,
        cost,
        (price_norm - cost) AS unit_margin,
        CASE WHEN price_norm = 0 THEN NULL ELSE (price_norm - cost)/price_norm END AS margin_pct
      FROM cleaned
    )
    SELECT * FROM metrics
    """

    dim_prod = spark.sql(sql)

    out = "s3a://my-bucket/silver/dim_product/"
    dim_prod.write.mode("overwrite").parquet(out)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("dim_product_sql_cte").getOrCreate()
    run(spark)


