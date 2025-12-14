from pyspark.sql import functions as F

def run(spark):
    orders = spark.read.parquet("s3a://my-bucket/gold/fact_orders_enriched/")
    risk = spark.read.parquet("s3a://my-bucket/gold/fact_orders_risk_scored/")
    cust = spark.read.parquet("s3a://my-bucket/silver/dim_customer/")

    orders.createOrReplaceTempView("orders")
    risk.createOrReplaceTempView("risk")
    cust.createOrReplaceTempView("cust")

    sql = """
    WITH base AS (
      SELECT
        o.customer_id,
        o.order_date,
        o.abs_amount,
        o.is_refund,
        COALESCE(r.raw_risk_score, 0) AS raw_risk_score,
        COALESCE(r.risk_bucket, 'unknown') AS risk_bucket
      FROM orders o
      LEFT JOIN risk r
        ON o.order_id = r.order_id
    ),
    daily AS (
      SELECT
        customer_id,
        order_date,
        SUM(abs_amount) AS daily_spend,
        SUM(CASE WHEN is_refund = 1 THEN abs_amount ELSE 0 END) AS daily_refund_amt,
        MAX(raw_risk_score) AS max_risk_score_day,
        MAX(CASE WHEN risk_bucket IN ('critical','high') THEN 1 ELSE 0 END) AS any_high_risk_day
      FROM base
      GROUP BY customer_id, order_date
    ),
    customer_rollup AS (
      SELECT
        customer_id,
        SUM(daily_spend) AS total_spend,
        SUM(daily_refund_amt) AS total_refunds,
        AVG(daily_spend) AS avg_daily_spend,
        MAX(max_risk_score_day) AS max_risk_score,
        MAX(any_high_risk_day) AS ever_high_risk
      FROM daily
      GROUP BY customer_id
    )
    SELECT
      c.customer_id,
      c.customer_tier,
      cr.total_spend,
      cr.total_refunds,
      (cr.total_spend - cr.total_refunds) AS net_spend,
      cr.avg_daily_spend,
      cr.max_risk_score,
      cr.ever_high_risk,
      CASE
        WHEN cr.total_spend >= 10000 AND cr.ever_high_risk = 0 THEN 'VIP_SAFE'
        WHEN cr.total_spend >= 10000 AND cr.ever_high_risk = 1 THEN 'VIP_RISK'
        WHEN cr.total_spend >= 2000 THEN 'HIGH_VALUE'
        ELSE 'STANDARD'
      END AS customer_segment
    FROM cust c
    LEFT JOIN customer_rollup cr
      ON c.customer_id = cr.customer_id
    """

    mart = spark.sql(sql)

    out = "s3a://my-bucket/gold/mart_customer_value/"
    mart.write.mode("overwrite").parquet(out)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("mart_customer_value").getOrCreate()
    run(spark)



