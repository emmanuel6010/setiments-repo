# hokum

*Analyze sentiments on text*

Copyright 2023 Emmanuel Asante

# Function Reference

```{ .python silent }

from hokum import text_sentiment, word_cloud, common_words_data, common_words_text, sentiment_graph, word_cloud_dataframe, sentiment_data, merge_dataframes, read_data_file, recommend_data, audio_transcription

from npdoc_to_md import render_md_from_obj_docstring

print(render_md_from_obj_docstring(read_data_file, 'hokum.read_data_file'))
print('\n\n')

print(render_md_from_obj_docstring(merge_dataframes, 'hokum.merge_dataframes'))
print('\n\n')

print(render_md_from_obj_docstring(sentiment_data, 'hokum.sentiment_data'))
print('\n\n')

print(render_md_from_obj_docstring(word_cloud_dataframe, 'hokum.word_cloud_dataframe'))
print('\n\n')

print(render_md_from_obj_docstring(sentiment_graph, 'hokum.sentiment_graph'))
print('\n\n')

print(render_md_from_obj_docstring(common_words_data, 'hokum.common_words_data'))
print('\n\n')

print(render_md_from_obj_docstring(recommend_data, 'hokum.recommend_data'))
print('\n\n')

print(render_md_from_obj_docstring(text_sentiment, 'hokum.text_sentiment'))
print('\n\n')

print(render_md_from_obj_docstring(word_cloud, 'hokum.word_cloud'))
print('\n\n')

print(render_md_from_obj_docstring(common_words_text, 'hokum.common_words_text'))
print('\n\n')

print(render_md_from_obj_docstring(audio_transcription, 'hokum.audio_transcription'))
print('\n\n')

```