# AI_Test - Sentiment Analysis

==========================================

# Libraries and Models Used
---------------------------------


1. Libraries: 
  - torch
  - transformers
  - nltk
  - matplotlib

2. Models: 
  - RoBERTa (for sentiment analysis)

============================================================================================================

# Reading Reviews
---------------------

The reviews are read from a text file named 'hotel_reviews.txt'.
Each review is processed line by line.

Code Snippet:
```python
file_path = 'hotel_reviews.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    reviews = file.readlines()
```

===========================================================================================================

# **Sentiment Analysis**
---------------------------

We use the pre-trained RoBERTa model for sentiment analysis.
The sentiment of each sentence is categorized as Positive, Neutral, or Negative.

Code Snippet:
```python
def sentiment_analysis(text):
  inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
  outputs = roberta_model(**inputs)
  scores = outputs.logits.detach().numpy()[0]
  sentiment = scores.argmax()
  if sentiment == 0:
      return "NEGATIVE"
  elif sentiment == 1: 
      return "NEUTRAL"
  else: 
      return "POSITIVE"

```

============================================================================================================

# **Visualizing Sentiment Distribution**
----------------------------------------


We use the `matplotlib` library to visualize the sentiment distribution.
A pie chart is created to show the percentage of positive, neutral, and negative reviews.

Code Snippet:
```python
labels = list(sentiment_counts.keys())
sizes = list(sentiment_counts.values())
colors = ['green', 'blue', 'red']
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, sizes, color=colors)
# Add text annotations on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Distribution of Hotel Reviews')
plt.show()

==============================================================================================================

# **Conclusion**
-------------

The sentiment analysis and visualization provide insights into customer feedback.
Positive reviews indicate areas of strength, while negative and neutral reviews highlight areas for improvement.
![image](https://github.com/yuga2005/AI_Test/assets/13882017/123f9d7c-0d8a-4f4a-a5b5-d3eaf2b555f4)

