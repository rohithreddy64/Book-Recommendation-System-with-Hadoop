from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Book Recommendation System") \
    .getOrCreate()

# Read data from CSV into dataframes
df_BX_Users = spark.read.options(delimiter=";", header="true").csv("hdfs://localhost:9000/data-lake/BX-Users.csv")
df_BX_Books = spark.read.options(delimiter=";", header="true").csv("hdfs://localhost:9000/data-lake/BX-Books.csv")
df_BX_Book_Ratings = spark.read.options(delimiter=";", header="true").csv("hdfs://localhost:9000/data-lake/BX-Book-Ratings.csv")

# Convert 'Book-Rating' column to integer
df_BX_Book_Ratings = df_BX_Book_Ratings.withColumn('Book-Rating', df_BX_Book_Ratings['Book-Rating'].cast(IntegerType()))

# Top 10 Publisher
Top_10_Publisher = df_BX_Books.groupBy("Publisher").count().sort('count', ascending=False).limit(10)

# Year trend of book publishing
Year_Trend_Publishing = df_BX_Books.groupBy("Year-Of-Publication","Publisher").count().sort('count', ascending=False).limit(20)

# Join Three tables
df_join = df_BX_Books.join(df_BX_Book_Ratings, on=['ISBN'], how='inner')
df_join_2 = df_join.join(df_BX_Users, on=['User-ID'], how='inner')

# Number of users per book
Max_users_per_bookID = df_join.groupBy('ISBN', 'Book-Title').agg(countDistinct('User-ID')).sort('count(DISTINCT User-ID)', ascending=False).limit(10)

# Average rating of a book
Avg_BookID_Rating = df_join.groupBy('ISBN').avg('Book-Rating').sort('avg(Book-Rating)', ascending=False)

# Compute count of ratings for each BookID
counts = df_join.groupBy('ISBN').count().sort('count', ascending=False)

# Join the two together (We now have ISBN, avg(rating), and count columns)
AvgCount = counts.join(Avg_BookID_Rating, 'ISBN').sort('avg(Book-Rating)', ascending=False).limit(10)

# Top 10 book names
Top_BookNames = AvgCount.join(df_BX_Books, on=['ISBN'], how='inner').limit(10).select("Book-Title")

# Recommendation

# Transformation
ratings = df_join.select('ISBN', 'User-ID', 'Book-Rating').filter(df_join["User-ID"].isNotNull() & df_join["ISBN"].isNotNull())
ratings = ratings.withColumn('ISBN', ratings['ISBN'].cast(IntegerType()))
ratings = ratings.withColumn('User-ID', ratings['User-ID'].cast(IntegerType()))

# Split data into test and train
train = ratings.sort('ISBN', ascending=False).limit(2000)

# Find books rated more than 10 times
ratingCounts = train.groupBy("ISBN").count().filter("count > 10")

# Construct a "test" dataframe for user 0 with every movie rated more than 100 times
#test = ratingCounts.select("ISBN").withColumn('User-ID', lit(14232)).join(df_BX_Book_Ratings, on=['ISBN'], how='inner')
# Construct a "test" dataframe for user 0 with every movie rated more than 100 times

#test = ratingCounts.select("ISBN").withColumn('User-ID', lit(14232)).join(df_BX_Book_Ratings, on=['ISBN'], how='inner')
# Join the dataframes and alias the literal value 14232
test = ratingCounts.select("ISBN").withColumn('User-ID-Lit', lit(14232)).join(df_BX_Book_Ratings, on=['ISBN'], how='inner')

test.show()
# Create an ALS collaborative filtering model
from pyspark.ml.recommendation import ALS
als = ALS(rank=25, maxIter=5, regParam=0.09, userCol="User-ID", itemCol="ISBN", ratingCol="Book-Rating", nonnegative=True)
model = als.fit(train)
test = test.withColumn('User-ID', test['User-ID'].cast(IntegerType()))
# Run model on the list of popular books for user ID 14232
recommendations = model.transform(test)

# Get the top 20 books with the highest predicted rating for this user
topRecommendations = recommendations.sort(recommendations.prediction.desc()).take(20)
topRecommendations = spark.createDataFrame(topRecommendations)

# Show top recommendations
topRecommendations.join(df_BX_Books, on=['ISBN'], how='inner').sort('prediction', ascending=False).limit(10).select("Book-Title", "prediction").show()

# Extracting data

# Filter books published in 1999
df_BX_Books_filtered = df_BX_Books.filter(df_BX_Books["Year-Of-Publication"].contains('1999'))
#df_BX_Books_filtered.write.format('com.databricks.spark.csv').save('prigupt/book_join_1999.csv', header='true')
# Filter books published by 'Zebra Books'
df_BX_Books_filtered = df_BX_Books.filter(df_BX_Books["Publisher"].contains('Zebra Books'))
#df_BX_Books_filtered.write.format('com.databricks.spark.csv').save('prigupt/book_join_pub.csv', header='true')
# Filter books authored by 'Agatha Christie'
df_BX_Books_filtered = df_BX_Books.filter(df_BX_Books["Book-Author"].contains('Agatha Christie'))
#df_BX_Books_filtered.write.format('com.databricks.spark.csv').save('prigupt/book_join_auth.csv', header='true')
# Filter book ratings for a specific ISBN
df_BX_Books_filtered = df_BX_Book_Ratings.filter(df_BX_Book_Ratings["ISBN"].contains('0971880107'))
#df_BX_Books_filtered.write.format('com.databricks.spark.csv').save('prigupt/book_join_rat.csv', header='true')