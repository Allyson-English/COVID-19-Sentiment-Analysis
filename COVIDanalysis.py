import numpy as np
# import textblob as textblob
import praw
from datetime import datetime
import json
import creds
from pprint import pprint
import pandas as pd
from time import time

#Importing NLTK library and associated packaged

import nltk
nltk.__version__
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag_sents
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Importing libraries for visualization 

import plotly as py
import plotly.graph_objs as go
print(py.__version__)
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Reddit API wrapper 

reddit = praw.Reddit(client_id=creds.client_id, \
                     client_secret=creds.client_secret, \
                     user_agent=creds.user_agent, \
                     username=creds.username, \
                     password=creds.password)

subreddit = reddit.subreddit('Coronavirus')

path = '***'

with open(path, "r") as read_file:
    data = json.load(read_file)
    
# Pulls all discussion IDS from specified subreddit
# Note that default parameters are for all Daily Discussion Posts 
# To specify a date range include 'Month Date'
# To specif a new search term include 'New search criteria' (must come after date)


def get_comments(sub, existing_dict, date='', search_term='Daily Discussion Post'):
    
    full_search = search_term + " - " + date
    count = 0
    
    for submission in subreddit.search(full_search):
        if search_term in submission.title:
            count+=1
            if submission.id in existing_dict.keys():
                pass
            else:
                existing_dict.update({submission.id:{}})
    print(count, "discussion ids added to dictionary.")
                
    return existing_dict
    
data = get_comments(subreddit, data)


def comment_info(us_com, submission, sid):
    
    token_dict = {}
    
    comment = us_com.body
    comment = comment.replace('\n', ' ')
    comment = comment.replace('I\'m', 'i am').replace('i\'m', 'i am').replace('i\'ll', 'i will').replace('I\'ll', 'i will')
    comment = comment.lower()
    
    #grab date/ time info for each comment 
    utc = submission.created_utc
    dt_object = datetime.fromtimestamp(utc)  

    #performing sentiment analysis
    ss = sid.polarity_scores(comment)
    
    if comment != '[removed]':
        token_dict.update({"comment_body" : comment})
        token_dict.update({"month":dt_object.strftime("%B")})
        token_dict.update({"day" : dt_object.strftime("%d")})
        token_dict.update(ss)
    
    return token_dict


#pulls everything together and returns complete dataset in the form of a dictionary

def nltk_sentiment(existing_dict, reddit = reddit):

    sid = SentimentIntensityAnalyzer()
    st = time()
    tok_dict = {}
    count = 0
    
    for i in existing_dict.keys():
        count+= 1
        print("Fetching comments for Daily Discussion: ", i, " ", count, "/", len(existing_dict.keys()))
        existing_dict[i] = {}
        submission = reddit.submission(i)
        submission.comments.replace_more(limit=0)

        for user_comment in submission.comments:
            tok_dict = comment_info(user_comment, submission, sid)

            if str(user_comment) not in existing_dict[i].keys():
                existing_dict[i].setdefault(str(user_comment),tok_dict)
        
    print("\nProcessing time:", round((time()-st)/60, 2), "minutes.")
    
    return existing_dict
    
    
# Conduct sentiment analysis
# Note that the sentiment analysis for every comment takes some time becaue 
# it re-does the analysis on older ones, this will have to be sorted out going forward.

# this takes ~12 minutes to run

updated_dict = nltk_sentiment(data)

# checks if there are missing values for keys in dictionary and removes them

def remove_missing_values(dictionary):
    """Comments that have been removed by moderators return empty values in previous function. 
    This function removes missing values."""
    
    missing = []
    d = dictionary.copy()

    for k in dictionary.keys():
        for a in dictionary[k].keys():
            if len(dictionary[k][a]) == 0:
                missing.append((k, a))
    
    for di, ci in missing:
        d[di].pop(ci)
    
    print(len(missing), "missing values have been removed from this dictionary.")
    
    return d
    
    
updated_dict = remove_missing_values(updated_dict)

with open(path, 'w') as write_file:
    json.dump(updated_dict, write_file)
    
    
# Data Visualization
# open data file if not already active in cell 

with open(path, "r") as read_file:
    data = json.load(read_file)
    
    
discussion_id = data.keys()

m_date = []
d_date = []
negative = []
neutral = []
positive = []
compound = []
comment_id = []
discussion = []

for k in data.keys():
    
    for y in data[k].keys():
        discussion.append(k)
        comment_id.append(y)
    
    for x in data[k]:
        m_date.append(data[k][x].get('month'))
        d_date.append(data[k][x].get('day'))
        negative.append(data[k][x].get('neg'))
        neutral.append(data[k][x].get('neu'))
        positive.append(data[k][x].get('pos'))
        compound.append(data[k][x].get('compound'))
        
d = {'discussion_id' : discussion,'comment_id' : comment_id, 'month': m_date, 'date': d_date, 'positive': positive, 'neutral': neutral, 'negative': negative, 'compound': compound}
df = pd.DataFrame(data=d)
df.head()


def sort_days(x):
    if x.startswith('F'):
        return 2
    elif x.startswith('Mar'):
        return 3
    elif x.startswith('April'):
        return 4
    elif x.startswith('May'):
        return 5
    
def sent_classification(x):
    if x <= -0.5:
        return "Strong Negative"
    if -0.5 < x <=0:
        return "Negative"
    if 0 < x < 0.5:
        return "Neutral"
    if x >=0.5:
        return "Positive"
    
df['DOM'] =df['month'] + " " + df['date']

df2 = df.set_index(["DOM", "comment_id"]).count(level="DOM").copy()
df2 = df2.reset_index()

df2['M'] = df2['DOM'].apply(sort_days)
df2['Sent_Class'] = df2['compound'].apply(sent_classification)

df2.sort_values(['M', 'DOM'], inplace = True)

df2 = df2.reset_index()

df2.head()


daily_avrg = df.groupby('DOM').mean().reset_index()

daily_avrg['M'] = daily_avrg['DOM'].apply(sort_days)
daily_avrg['Sent_Class'] = daily_avrg['compound'].apply(sent_classification)

daily_avrg.sort_values(['M', 'DOM'], inplace = True)

daily_avrg = daily_avrg.reset_index()

daily_avrg.head()

df3 = daily_avrg.merge(df2, left_on='DOM', right_on='DOM')
df3.head()

trace = go.Scatter(x=daily_avrg['DOM'],
                  y=round(daily_avrg['compound'], 4),
                   y0='Negative',
                  mode='markers',
                   name= "",
                  marker = dict(size=df3['discussion_id']/6, color=daily_avrg['compound'],
                                colorscale = 'Portland_r', showscale = True)
                  )

z = np.polyfit(daily_avrg.index, daily_avrg['compound'], 1)
p = np.poly1d(z)

trace1 = go.Scatter(x=daily_avrg['DOM'], y=p(daily_avrg.index),
                    mode='lines',
                    name='Trendline',
                    line=dict(color='black'))

data = [trace, trace1]

layout = {
    "title": "Online COVID-19 Sentiment: February Through May 2020",
    'title_x': 0.5,
    "xaxis": {
        "title": "Day",
        "zeroline": True,
#         "tickangle": 0,
#         'ticktext': [
#             'February',
#             'March',
#             'April',
#             'May'
#         ],
#         'tickvals': [
#             7,
#             25,
#             37,
#             42
#         ]
    },
    "yaxis": {
        "title": "Sentiment Score",
        "zeroline": True,
        'zerolinecolor': 'grey',
        'zerolinewidth': .169,
        'tickmode': 'array',
        'ticktext': [
            'Negative  ',
            'Neutral  ',
            'Positive  '
        ],
        'tickvals': [
            -.16,
            0,
            .12
        ]
    },
    "showlegend": False,
    'autosize': False,
    'width': 1300,
    'height': 700,
    'paper_bgcolor': 'rgb(0,0,0, 0)',
    'plot_bgcolor': 'rgb(0,0,0, 0)',
}



plot({"data": data, "layout": layout}, output_type='file', include_plotlyjs=True, show_link = True)

h = py.io.to_html({"data": data, "layout": layout}, include_mathjax=False, full_html=False)

with open('/Users/allysonenglish/Desktop/test3sent.html', 'w') as out:
    out.write(h)
    
with open('/Users/allysonenglish/Desktop/smallsenttext.txt', 'w') as out:
    out.write(h)
