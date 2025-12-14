from pyspark.sql import functions as F, Window

def run(spark):
    clicks = spark.read.parquet("s3a://my-bucket/raw/web_clicks/")  # customer_id, event_ts, url, referrer

    df = (clicks
          .withColumn("event_ts", F.to_timestamp("event_ts"))
          .withColumn("event_date", F.to_date("event_ts"))
          .withColumn("url_host", F.regexp_extract(F.col("url"), r"https?://([^/]+)/", 1))
          .withColumn("ref_host", F.regexp_extract(F.col("referrer"), r"https?://([^/]+)/", 1))
    )

    w = Window.partitionBy("customer_id").orderBy("event_ts")
    df2 = df.withColumn("prev_ts", F.lag("event_ts").over(w))
    df3 = df2.withColumn("gap_min",
                         (F.col("event_ts").cast("long") - F.col("prev_ts").cast("long")) / 60.0)

    df4 = df3.withColumn("new_session_flag",
                         F.when((F.col("gap_min").isNull()) | (F.col("gap_min") > 30), 1).otherwise(0))

    df5 = df4.withColumn("session_num", F.sum("new_session_flag").over(w))
    df6 = df5.withColumn("session_id", F.concat_ws("-", F.col("customer_id"), F.col("event_date"), F.col("session_num")))

    df6.createOrReplaceTempView("sessions")

    agg_sql = """
      SELECT
        customer_id,
        session_id,
        MIN(event_ts) AS session_start_ts,
        MAX(event_ts) AS session_end_ts,
        COUNT(1) AS events,
        COUNT(DISTINCT url_host) AS unique_hosts
      FROM sessions
      GROUP BY customer_id, session_id
    """

    sess_agg = spark.sql(agg_sql)

    out = "s3a://my-bucket/gold/web_sessions/"
    sess_agg.write.mode("overwrite").partitionBy("customer_id").parquet(out)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("web_sessionization").getOrCreate()
    run(spark)



