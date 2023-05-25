# Import libraries
import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.parsing.preprocessing import remove_stopwords
import speech_recognition as sr
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence



# join text column for a dataframe
def join_words(dataframe, column):
    return ' '.join([word for word in dataframe[column]])



# remove special characters
def remove_special_char(s):
    return re.sub(r'[^a-zA-Z0-9\s]','',s)



# Function to tokenize a column of a dataframe
# The function tokenization
def tokenization(dataframe, column):
    regexp = RegexpTokenizer('\w+')
    dataframe['tokenized_review'] = dataframe[column].apply(regexp.tokenize)
    return dataframe


################################################################################################################################
################################################################################################################################
################################################################################################################################



def text_sentiment(text):
    """Analyze sentiments on a given text
    
    Parameters
    ----------
    
    text: str
               The given text
               
    Returns
    -------
    str
               Analyze sentiment on the text
    
    """
    analyzer = SentimentIntensityAnalyzer()
    polarity = analyzer.polarity_scores(text)
    if polarity['compound'] > 0:
        return str(round(polarity['compound']*100,1)) + '% ' + 'Positive'
    elif polarity['compound'] < 0:
        return str(abs(round(polarity['compound']*100,1))) + '% ' + 'Negative'
    else:
        return 'Neutral'
    
    





def word_cloud(text, width=600, height=400, random_state=2, max_font_size=100, background_color='white', fig_width=10, fig_height=7):
    """Generate word cloud for a given text.
    
    Parameters
    ----------
    
    text: str
               The text
               
    width: int, default=600
               The width of the word cloud.
               
    height: int, default=400
               The height of the word cloud.
               
    random_state: int, default=2
               Minimum number of letters a word must have to be included.
               
    max_font_size: int, default=100
               Maximum font size for the largest word.
               
    background_color: str, default='white
               Background color for the word cloud.
               
    fig_width: int, default=10
               The width of the figure.
               
    fig_height: int, default=7
               The height of the figure.
    
    Returns
    ------
    Word Cloud
               Generate a word cloud
    """
    
    text = remove_special_char(text)
    text = text.lower()
    tokenize_overall_words = nltk.tokenize.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words_lemmatize = []
    for word in tokenize_overall_words:
        words_lemmatize.append(lemmatizer.lemmatize(word))
    all_words = ' '.join(words_lemmatize)
    stopwords = nltk.corpus.stopwords.words('english')
    wordcloud = WordCloud(width=width, 
                         height=height, 
                         random_state=random_state, 
                         max_font_size=max_font_size,
                         stopwords=stopwords,
                         collocations=False,
                         background_color=background_color).generate(all_words)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off');
    
    
    
# graph most common words from a dataframe
def common_words_data(dataframe, column, num=10):
    """Graph the most common words from a text column in a data frame.
    
    Parameters
    ----------
    
    dataframe: DataFrame
               The dataset.
    
    column: str
               The text column of the dataset.
               
    num: int, default=10
               The top n words from the text.
               
    Returns
    -------
    Bar Chart
               Displays a barplot for most common words from the text.
    
    """
    all_words = join_words(dataframe, column)
    word = remove_special_char(all_words)
    words = remove_stopwords(word)
    tokens = nltk.word_tokenize(words)
    frequency_dist = FreqDist(tokens)
    top_num = frequency_dist.most_common(num)
    frequency_dist_series = pd.Series(dict(num))
    sns.set_theme(style="ticks")
    sns.barplot(y=frequency_dist_series.index, x=frequency_dist_series.values, color='green');
    
    
    
    

def common_words_text(text, num=10):
    """Counts common words from a text and returns a bar graph of the top n words from the text.
    
    Parameters
    ----------
    
    text: str
               The text
               
    num: int, default=10
               The top n words from the text.
               
    Returns
    -------
    Bar Chart
               Displays a barplot for most common words from the text.
    
    """
    word = remove_special_char(text)
    words = remove_stopwords(word)
    tokens = nltk.word_tokenize(words)
    frequency_dist = FreqDist(tokens)
    top_num = frequency_dist.most_common(num)
    frequency_dist_series = pd.Series(dict(num))
    fig = px.bar(y=frequency_dist_series.index, x=frequency_dist_series.values, labels={"y": "Words", "x": "Counts"})
    fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    fig.show()
    
    
    
    
def sentiment_graph(dataframe, column):
    """Displays a bar graph for sentiment labels from a data frame.
    
    Parameters
    ----------
    
    dataframe: DataFrame
               The data frame.
               
    column: str
               The column name which contains the sentiment labels.
               
    Returns
    -------
    CountPlot
               Display a graph across sentiment labels in the data frame.
    
    
    """
    sns.countplot(y=column, 
             data=dataframe, 
             palette=['#b2d8d8',"#008080", '#db3d13']
             );
    
    
    
    
# generate word cloud from a dataframe given a specified column    
def word_cloud_dataframe(dataframe, column, width=600, height=400, random_state=2, max_font_size=100, background_color='white', fig_width=10, fig_height=7):
    """Generate word cloud from a data frame given a specified column.
    
    Parameters
    ----------
    
    dataframe: DataFrame
               The data frame.
            
    column: str
               The selected column from the data frame.
               
    width: int, default=600
               The width of the word cloud.
               
    height: int, default=400
               The height of the word cloud.
               
    random_state: int, default=2
               Minimum number of letters a word must have to be included.
               
    max_font_size: int, default=100
               Maximum font size for the largest word.
               
    background_color: str, default='white'
               Background color for the word cloud.
               
    fig_width: int, default=10
               The width of the figure.
               
    fig_height: int, default=7
               The height of the figure.
    
    Returns
    ------
    Word Cloud
               Generate a word cloud
    """
    
    dataframe[column] = dataframe[column].astype(str).str.lower()
    df = dataframe
    regexp = RegexpTokenizer('\w+')
    df['tokenized_review'] = dataframe[column].apply(regexp.tokenize)
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(['https', 'jpeg', 'jpg', 'mp3', 'mp4'])
    df['tokenized_review'] = df['tokenized_review'].apply(lambda x: [item for item in x if item not in stopwords])
    df['review_string'] = df['tokenized_review'].apply(lambda x: ' '.join([item for item in x if len(item)>1]))
    overall_words = ' '.join([word for word in df['review_string']])
    tokenize_overall_words = nltk.tokenize.word_tokenize(overall_words)
    word_frequency = FreqDist(tokenize_overall_words)
    df['review_string_frequency'] = df['tokenized_review'].apply(lambda x: ' '.join([item for item in x if word_frequency[item] >= 1 ]))
    lemmatizer = WordNetLemmatizer()
    df['review_string_lemmatized'] = df['review_string_frequency'].apply(lemmatizer.lemmatize)
    all_words_lemmatized = ' '.join([word for word in df['review_string_lemmatized']])
    wordcloud = WordCloud(width=width, 
                     height=height, 
                     random_state=random_state, 
                     max_font_size=max_font_size,
                     collocations=False,
                     background_color=background_color).generate(all_words_lemmatized)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off');
    
    
    
    

def sentiment_data(dataframe, column):
    """Generate sentiment score for a given dataset and a selected column.
    
    Parameters
    ----------
    
    dataframe: DataFrame
               The given data
               
    column: str
               The selected column to generate sentiment scores for.
    
    Returns
    -------
    DataFrame
               The data frame with sentiment scores and tags.
    """
    
    dataframe[column] = dataframe[column].astype(str).str.lower()
    df = dataframe
    regexp = RegexpTokenizer('\w+')
    df['tokenized_review'] = dataframe[column].apply(regexp.tokenize)
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(['https', 'jpeg', 'jpg', 'mp3', 'mp4'])
    df['tokenized_review'] = df['tokenized_review'].apply(lambda x: [item for item in x if item not in stopwords])
    df['review_string'] = df['tokenized_review'].apply(lambda x: ' '.join([item for item in x if len(item)>1]))
    overall_words = ' '.join([word for word in df['review_string']])
    tokenize_overall_words = nltk.tokenize.word_tokenize(overall_words)
    word_frequency = FreqDist(tokenize_overall_words)
    df['review_string_frequency'] = df['tokenized_review'].apply(lambda x: ' '.join([item for item in x if word_frequency[item] >= 1 ]))
    lemmatizer = WordNetLemmatizer()
    df['review_string_lemmatized'] = df['review_string_frequency'].apply(lemmatizer.lemmatize)
    analyzer = SentimentIntensityAnalyzer()
    df['polarity'] = df['review_string_lemmatized'].apply(lambda x: analyzer.polarity_scores(x))
    df = pd.concat(
    [df.drop(['polarity'], axis=1),
    df['polarity'].apply(pd.Series)
    ], axis=1)
    df['sentiment_'+ column] = df['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
    df.drop(columns=['tokenized_review', 'review_string', 'review_string_frequency', 'review_string_lemmatized', 'neg', 'neu', 'pos'], inplace=True)
    df.rename(columns={'compound':'sentiment_score'}, inplace=True)
    return df







def merge_dataframes(dataframes, on=None, how='inner'):
    """Merges a list of data frames and returns the resulting data frame.
    
    Parameters
    ----------
    
    dataframes: DataFrame
                List of data frames
                
    on: label, list, default=None
                Column or index level names to join on. These must be found in both DataFrames. If on is None and not merging on indexes then this defaults to the intersection of the columns in both DataFrames.
                
    how: {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’
                Type of merge to be performed.
                1. left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
                2. right: use only keys from right frame, similar to a SQL right outer join; preserve key order.
                3. outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
                4. inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
                5. cross: creates the cartesian product from both frames, preserves the order of the left keys.
    
    
    """
    merged_df = dataframes[0]
    for i in range(1, len(dataframes)):
        merged_df = pd.merge(merged_df, dataframes[i], on=on, how=how)
    return merged_df





def read_data_file(file_path, file_type):
    """Reads any data in csv or xlsx or json or html or sql file format and returns a data frame.
    
    Parameters
    ----------
    
    file_path: /file_path
               The location of the file.
               
    file_type: __.type
               The file type.
               
    Returns
    -------
    DataFrame
                The merged data frame.
    """
    
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_type == 'json':
        df = pd.read_json(file_path)
    elif file_type == 'html':
        dfs = pd.read_html(file_path)
        df = dfs[0]  # assumes the first table is the desired dataframe
    elif file_type == 'sql':
        sql_engine = create_engine('sqlite:///' + file_path)
        df = pd.read_sql_table(file_path, sql_engine)
    else:
        raise ValueError(f'File type "{file_type}" not supported.')

    return df





def recommend_data(dataframe, user_info, unique_id, *args):
    """Build a recommender system.
    
    Parameters
    ----------
    
    dataframe: DataFrame
               The data must contain a `sentiment_score` column.
            
    user_info: str or int
               The column with an information about a user, e.g. user_id, etc..
               
    unique_id: str or int
               A unique id for the user_info
               
    *args: str or int
               Addition column(s) from the data
               
    Returns
    -------
    DataFrame
               Generate sample random rows and column(s) from the data
               
    """
    # group similar text
    positives = dataframe[dataframe['sentiment_score']>0]
    negatives = dataframe[dataframe['sentiment_score']<0]
    neutrals = dataframe[dataframe['sentiment_score']==0]
    # create user profile
    positive_users = positives[user_info].tolist()
    negative_users = negatives[user_info].tolist()
    neutral_users = neutrals[user_info].tolist()
    # build a recommender system
    if unique_id in positive_users:
        return positives.sample()[[user_info]+list(args)]
    elif unique_id in negative_users:
        return negatives.sample()[[user_info]+list(args)]
    elif unique_id in neutral_users:
        return neutrals.sample()[[user_info]+list(args)]
    else:
        return 'User not found'
      
        
        
        

        
        

def audio_transcription(path):
    """Splits audio file into chunks and applies speech recognition.
    
    Parameter
    ---------
    
    path: __file
               The path to the audio file
               
    Returns
    -------
    str
               Provides a text from the audio
    
    """
    # create a speech recognition object
    r = sr.Recognizer()
    
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text
