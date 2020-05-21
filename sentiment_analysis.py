import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# clean text
text=open('jeff_bezos_speech.txt',encoding='utf-8').read()
# lowercase conversion
lower_case=text.lower()
# removing punctionations
cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation)).replace("\n",'')

# tokenization
tokenized_words=word_tokenize(cleaned_text,"english")
# removing stop words
final_words = []
for word in tokenized_words:
	if word not in stopwords.words('english'):
		final_words.append(word)

# emotion analysis
emotion_list=[] 
with open('emotions.txt','r') as file:
    for line in file:
        clear_line=line.replace("\n",'').replace(",",'').replace("'",'').strip()
        word,emotion = clear_line.split(':')
        if word in final_words:
            emotion_list.append(emotion)

#printing the sentiment analysed result
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    return score

result = sentiment_analyse(cleaned_text)

# plotting the emotions

n = Counter(emotion_list)
figs,ax2 = plt.subplots()
ax2.set_title('Emotional Analysis')
ax2.bar(n.keys(),n.values())
figs.autofmt_xdate() 
plt.show()

# plotting the sentiments
fig,ax1=plt.subplots()
ax1.set_title('Sentiment Analysis')
result.pop('compound')
ax1.bar(result.keys(),result.values())
fig.autofmt_xdate() 
plt.show()