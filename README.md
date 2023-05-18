Sentiment Analysis On Product Reviews Using Lexicon Based Model (VADER)
==============================================================
![31422Click on Create (5)](https://github.com/emmanuel6010/setiments-repo/assets/76977423/a214e556-5bb5-4b91-a555-796de875316a)


### The project's objective is to comprehend reviewers' sentiments about a product.




## Sentiment Analysis:

* Sentiment analysis and opinion mining are the computational studies of user opinion to assess the social, psychological, philosophical, and behavioral behaviors and perceptions of a single person or a group of people about a good, service, policy, or certain scenarios. Sentiment analysis is a crucial area of study that helps with product decision-making by revealing people's feelings behind a text.


## Customer Product reviews:
* A customer review is a critique of a good or service written by a consumer who has used it or had some other interaction with it. Customer reviews on e-commerce and online buying platforms are a type of customer feedback. 90% of consumers read product reviews online before making a purchase, and 88% of them believe that product reviews are just as reliable as personal recommendations.


### Dataset
This dataset is having the data of 1K+ Amazon Product's Ratings and Reviews as per their details listed on the official website of Amazon

#### Features

* product_id - Product ID
* product_name - Name of the Product
* category - Category of the Product
* discounted_price - Discounted Price of the Product
* actual_price - Actual Price of the Product
* discount_percentage - Percentage of Discount for the Product
* rating - Rating of the Product
* rating_count - Number of people who voted for the Amazon rating
* about_product - Description about the Product
* user_id - ID of the user who wrote review for the Product
* user_name - Name of the user who wrote review for the Product
* review_id - ID of the user review
* review_title - Short review
* review_content - Long review
* img_link - Image Link of the Product
* product_link - Official Website Link of the Product

* Source: [Data](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)


# Workflow
![](SentiBlog_1.png);

### Libraries
```python
from hokum import text_sentiment, word_cloud, common_words_data, common_words_text, sentiment_graph, word_cloud_dataframe, sentiment_data, merge_dataframes, read_data_file, recommend_data, audio_transcription
```

### Steps
1. Load Data
```python
df = read_data_file('../data/raw/amazon.csv', 'csv')
df.head(20)
```
![df1](https://github.com/emmanuel6010/setiments-repo/assets/76977423/014aa82b-318b-4779-9868-a6b14bda979c)

2. Data Preprocessing and sentiment polarity scores for user reviews on the dataset using a rule-based model
   * Remove Punctuations,special symbols and special characters.

   * Stopword Removal

   * Tokenization

   * Lemmatization

   * VADER
```python
sentiments = sentiment_data(df, 'review_content')
sentiments.head(20)
```
![df2](https://github.com/emmanuel6010/setiments-repo/assets/76977423/7902774e-26a6-4619-95bc-076b1081822f)

3. Visualize the general sentiment labels
```python
sentiment_graph(sentiments, 'sentiment_review_content')
```
<img width="724" alt="Screenshot 2023-05-18 at 9 12 06 AM" src="https://github.com/emmanuel6010/setiments-repo/assets/76977423/621f040d-d086-4c80-99f5-0b8e8c66f011">

4. Word Cloud: Generate word cloud for the overall reviews (review_content) from the dataframe
```python
word_cloud_dataframe(df, 'review_content')
```
<img width="784" alt="Screenshot 2023-05-18 at 8 29 01 AM" src="https://github.com/emmanuel6010/setiments-repo/assets/76977423/9d5c64e2-8ac6-417c-96ab-94ea7546e569">

5. Count the top ten words
```python
common_words_data(df, 'review_content')
```
<img width="724" alt="Screenshot 2023-05-18 at 9 25 28 AM" src="https://github.com/emmanuel6010/setiments-repo/assets/76977423/5494ea7d-7076-449f-893b-a47f1db3b244">


## Function Reference
```python
import hokum
```

### ***hokum.read_data_file(file_path, file_type)***

Reads any data in csv or xlsx or json or html or sql file format and returns a data frame.


**<ins>Parameters:</ins>** <br />
* **file_path: file_path** <br />
                The location of the file
                                  
* **file_type: file** <br />
                 Any data file type that pandas can read
                                           
**<ins>Returns:</ins> DataFrame** <br />
                  A data frame
                  
                 



### ***hokum.sentiment_data(dataframe, column)***

Generate sentiment score for a given dataset and a selected column.

**<ins>Parameters:</ins>** <br />
* **dataframe: DataFrame** <br />
                  The dataframe
                                  
* **column: str** <br />
                  The selected column to generate sentiment scores for.
                                           
 **<ins>Returns:</ins> DataFrame** <br />
                   A new data frame with two new columns. A sentiment_score column containing sentiment scores and sentiment_column column containing the sentiment tag (Positive, Negative or Neutral).
                   
                   

### ***hokum.sentiment_graph(dataframe, column)***

Displays a bar graph for sentiment labels from a data frame.


**<ins>Parameters:</ins>** <br />
* **dataframe: DataFrame** <br />
                The data frame
                                  
* **column: str** <br />
                 The column name which contains the sentiment labels.
                                           
**<ins>Returns:</ins> CountPlot** <br />
                  Display a graph across sentiment labels in the data frame.
                  
                  
                  
                  
### ***hokum.word_cloud_dataframe(dataframe, column, width=600, height=400, random_state=2, max_font_size=100, background_color='white', fig_width=10, fig_height=7)***

Generate word cloud from a data frame given a specified column.


**<ins>Parameters:</ins>** <br />
* **dataframe: DataFrame** <br />
                The data frame.
                                  
* **column: str** <br />
                 The selected column from the data frame.
                 
* **width: int, default=600** <br />
                The width of the word cloud.
                
* **height: int, default=400** <br />
                The height of the word cloud.
                
* **random_state: int, default=2** <br />
                 Minimum number of letters a word must have to be included.
                 
* **max_font_size: int, default=100** <br />
                  Maximum font size for the largest word.
                  
* **background_color: str, default='white'** <br />
                  Background color for the word cloud.
                  
* **fig_width: int, default=10** <br />
                   The width of the figure.
* **fig_height: int, default=7** <br />
                   The height of the figure.
                                           
**<ins>Returns:</ins> Word Cloud** <br />
                  Generate a word cloud



### ***hokum.common_words_data(dataframe, column, num=10)***

Graph the most common words from a text column in a data frame.


**<ins>Parameters:</ins>** <br />
* **dataframe: DataFrame** <br />
                The available dataset
                                  
* **column: str** <br />
                 The text column of the dataset.
                 
* **num: int, default=10** <br />
                 The top 10 words from the text.              
                                           
**<ins>Returns:</ins> Bar Chart** <br />
                  Displays a barplot for most common words from the text.
                  
                  
                  
                  
                  
### ***hokum.text_sentiment(text)***

Analyze sentiments on a given text


**<ins>Parameter:</ins>** <br />
* **text: str** <br />
                A string text
                                               
                                           
**<ins>Returns:</ins> str** <br />
                  Analyze sentiment and prints out the sentiment score and tag on the text

**Example**
 ```python
>> text = 'pros- xiomi 5a is best in budget-nice picture quality-very nice audio output- full of featurecons- sometimes tv lags-sometimes stucksin this prize range all tv having cons like this.::overall nice tv,the product in this price range is good but as it is running in android 12 it lags. i hope after few updates the lags problem will be resolved,useless product and useless quality. display issues within 7 months and service center is not upto the mark. go for better brands where quality is assured. i would wish if there was option of negative stars.,uses as connectes tv the picture is very good. i was hopping a best level of song. globaly it is a good product.then ever,good quality,good 👍'
>> 
>> text_sentiment(text)
'96.2% Positive'
 ```
 
 
 
 

### ***hokum.word_cloud(text, width=600, height=400, random_state=2, max_font_size=100, background_color='white', fig_width=10, fig_height=7)***

Generate word cloud for a given text.


**<ins>Parameters:</ins>** <br />
* **text: string** <br />
                The text
                 
* **width: int, default=600** <br />
                The width of the word cloud.
                
* **height: int, default=400** <br />
                The height of the word cloud.
                
* **random_state: int, default=2** <br />
                 Minimum number of letters a word must have to be included.
                 
* **max_font_size: int, default=100** <br />
                  Maximum font size for the largest word.
                  
* **background_color: str, default='white'** <br />
                  Background color for the word cloud.
                  
* **fig_width: int, default=10** <br />
                   The width of the figure.
* **fig_height: int, default=7** <br />
                   The height of the figure.
                                           
**<ins>Returns:</ins> Word Cloud** <br />
                  Generate a word cloud
**Example**
```python
>> text1 = 'pros- xiomi 5a is best in budget-nice picture quality-very nice audio output- full of featurecons- sometimes tv lags-sometimes stucksin this prize range all tv having cons like this.::overall nice tv,the product in this price range is good but as it is running in android 12 it lags. i hope after few updates the lags problem will be resolved,useless product and useless quality. display issues within 7 months and service center is not upto the mark. go for better brands where quality is assured. i would wish if there was option of negative stars.,uses as connectes tv the picture is very good. i was hopping a best level of song. globaly it is a good product.then ever,good quality,good 👍'

>> word_cloud(text1)
```
<img width="846" alt="Screenshot 2023-05-18 at 2 53 46 PM" src="https://github.com/emmanuel6010/setiments-repo/assets/76977423/8b72c87e-d592-4f3e-b102-b7ccdb733a7c">
