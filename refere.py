# %% [markdown]
# <a href="https://colab.research.google.com/github/NamasteAI/ToxicCommentsClassifier/blob/main/ToxicCommentClassifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Import required libraries

# %%
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
#from google.colab import drive
#drive.mount('/content/drive')

# %%
#df = pd.read_csv('/content/drive/MyDrive/KaggleToxicCommentDataset/train.csv')
df = pd.read_csv('/Users/itscodezero/Documents/UOL/M6-NLP/MMA/data/train.csv')
df.head()

# %%
df.describe()

# %%
# Add a new column 'non-toxic' to the DataFrame with values 0 or 1, where 0 represent the comment falls under any one of the category
df['non-toxic'] = (df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)
df['non-toxic'] = df['non-toxic'].astype(int)  # Convert boolean to integer (True -> 1, False -> 0)

# %%
toxicity_counts = df.iloc[:, 2:].apply(pd.Series.value_counts)
print(toxicity_counts)

# %%
# Get the positive count of each category
category_totals = toxicity_counts.iloc[1].sort_values(ascending=False)
print(category_totals)

# %%

# Calculate the correlation matrix 
corr_matrix = df.iloc[:, 2:].corr()

# Plot the correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# %%

plt.figure(figsize=(20, 10))
bp = sns.barplot(x=category_totals.index, y=category_totals.values)
bp.set_yscale("log")
bp.tick_params(labelsize=15)

# %% [markdown]
# ## Dataset balancing

# %%
df_balanced = pd.concat([df[df['non-toxic'] == 1].sample(frac=0.11), df.loc[df['non-toxic'] == 0]])

# Shuffle the DataFrame
df_balanced = df_balanced.sample(frac=1)

toxicity_counts = df_balanced.iloc[:, 2:].apply(pd.Series.value_counts)
category_totals = toxicity_counts.iloc[1].sort_values(ascending=False)

plt.figure(figsize=(20, 10))
bp = sns.barplot(x=category_totals.index, y=category_totals.values)
bp.set_yscale("log")
bp.tick_params(labelsize=15)


# %% [markdown]
# # Preprocessing

# %%
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):

    # Normalize text (convert to lowercase)
    text = text.lower()
    
    # Initialize SpaCy
    doc = nlp(text)
    
    # Lemmatize and remove stopwords
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    
    # Join tokens back into text
    text = ' '.join(tokens)
    
    return text

print('Unprocessed text:', df_balanced['comment_text'].head())
# Apply preprocessing to the 'comment_text' column
df_balanced['comment_text'] = df_balanced['comment_text'].apply(preprocess_text)
print('Processed text:', df_balanced['comment_text'].head())

# %%
X = df_balanced['comment_text']
# take the columns from 2 to the 1 before the last one
y = df_balanced[df_balanced.columns[2:-1]].values
#print column names from y
print(df_balanced.columns[2:-1])

# %%
from tensorflow.keras.layers import TextVectorization

# %%
MAX_FEATURES = 200000

# %%
vectorizer = TextVectorization(max_tokens = MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')

# %%
vectorizer.adapt(X.values)

# %%
vectorizer('hello, how are you?')

# %%
vectorizer_text= vectorizer(X.values)

# %%
vectorizer_text

# %%
dataset = tf.data.Dataset.from_tensor_slices((vectorizer_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

# %%
batchX, batchy = dataset.as_numpy_iterator().next()

# %%
train = dataset.take(int(len(dataset)* .7))
val = dataset.skip(int(len(dataset)* .7)).take(int(len(dataset)* .2))
test = dataset.skip(int(len(dataset)* .9)).take(int(len(dataset)* .1))

# %% [markdown]
# ### Create Sequential Model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding

# %%
model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation= 'tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6,activation='sigmoid'))

# %%
model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer='Adam')

# %%
model.summary()

# %%
history = model.fit(train, epochs=1, validation_data=val)

# %%
model.evaluate(test)

# %%
history.history

# %%
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

# %% [markdown]
# # Evaluation metrics

# %%
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

# %%
pre = Precision()
re = Recall()
cat_acc = CategoricalAccuracy()

# %%

for batch in test.as_numpy_iterator():
    # Divide the batch into X_true and y_true
    X_true, y_true = batch

    # Make a prediction
    y_pred = model.predict(X_true)

    # flatten the prediction
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    pre.update_state(y_true, y_pred)
    re.update_state(y_true, y_pred)
    cat_acc.update_state(y_true, y_pred)


    

# %%
print (f'Precision {pre.result().numpy()}, Recall {re.result().numpy()}, Accuracy {cat_acc.result().numpy()}')

# %%
print(classification_report(y_true, y_pred))
print(f'AUC: {roc_auc_score(y_true, y_pred)}')


