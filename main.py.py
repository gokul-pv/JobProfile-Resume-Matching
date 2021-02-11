

import os
os.chdir("current path with resume")
import pandas as pd
import glob
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import math


# Bring in standard stopwords
stopWords = stopwords.words('english')

print ("\nCalculating document similarity scores...")

all_files = os.listdir("resume/")
print(all_files)
# Open and read a bunch of files
f = open(r'path to JD')
job_ds = str(f.read())
file_list = glob.glob(os.path.join(os.getcwd(), "resume", "*.txt"))
resume = []

for file_path in file_list:
    with open(file_path) as f_input:
        resume.append(str(f_input.read()))
resume.insert(0,job_ds)# tokenize all the documents
#print(resume)
def pre_process(text):
    text = text.replace(".", "") #remove full stops and commas
    text = text.replace(",", "")
    text= text.lower()#lower the text
    return text
processed_resume = [pre_process(file) for file in resume]

print(type(processed_resume))

# Set up the vectoriser, passing in the stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords)

# Apply the vectoriser to the with pre processing
tfidf_matrix_train = tfidf_vectorizer.fit_transform(processed_resume)


ss=cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)
ss=list(ss.flat)
resume_score=ss[1:]
#print(type(ss))
#print(ss)
#print(len(all_files))
# write into datframe
df=pd.DataFrame({"Resume":all_files,"Similarity score":resume_score},columns=['Resume','Similarity score'])
df=df.sort_values(['Similarity score'],ascending=False)


df.to_excel("resume_sheet.xlsx",index=False)
top_df=df[['Resume']][:2]
print("..................top 2 matching resumes are............... ")
print(top_df)
