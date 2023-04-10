import os
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import textdistance as tdist
import ast
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
sns.set()
os.chdir('..')

data = pd.read_csv('tokenized_lyrics.csv')
x = data['Lyric']
fig, ax = plt.subplots()
# # ax.set_xscale('log')
ax.set_xlabel('min document frequency')
# ax.set_yscale('log')
ax.set_ylabel('mean lyrical similarity')
mean_SL = []
std_SL = []
# Add the min_df range here
min_df_set = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
              0.08, 0.09, 0.1]
for min_df in min_df_set:
    vectorizer = TfidfVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(x.values.astype(str))
    tfidf_df = pd.DataFrame(
        X.toarray(), columns=vectorizer.get_feature_names_out())
    data = pd.concat([data, tfidf_df], axis=1)
    # Compute all pairwise similarities
    n = len(x.to_numpy())
    transformed = X.toarray()
    SL = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            SL[i, j] = np.dot(transformed[i], transformed[j]) / \
                (np.linalg.norm(transformed[i])
                 * np.linalg.norm(transformed[j]))
            SL[j, i] = SL[i, j]
    mean_SL.append(np.mean(SL))
    std_SL.append(np.std(SL))
ax.errorbar(min_df_set, mean_SL, yerr=std_SL, linewidth=2,
            capsize=5, capthick=2, elinewidth=2)
# add min df_set as x ticks
plt.xticks(min_df_set)

# also add as x labels
ax.set_xticklabels(min_df_set)


plt.plot()
plt.tight_layout()
plt.show()
