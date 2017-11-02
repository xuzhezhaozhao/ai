"""SimpleApp.py"""
from pyspark.sql import SparkSession

logFile = "ai_spam.txt"  # Should be some file on your system
spark = SparkSession.builder.appName("test").master("local[4]").getOrCreate()
logData = spark.read.text(logFile).cache()

numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

spark.stop()
