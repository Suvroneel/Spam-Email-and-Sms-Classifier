Spam sms Documentation 
It’s an end to end Ml project to detect spam messages
Classification problem
Modus Operand:
1.	Clean the data
a.	remove empty columns
b.	Label Encode spam and ham to 1 & 0 respectively
c.	Checking null values
d.	Checking duplicates and removing them

2.	Exploratory Data Analysis(EDA) :-Analyze the data
a.	Showing percentage of spam , not spam in pie chart
b.	We count no of alphabets,words,sentences in each sms and analyze that
c.	Use Histogram,Corr Coefficient to take decision
3.	Data preprocessing : Vectorization , other edits
1.    Lower Case
2.   Tokenization
3.   Removing Special Characters
4.   Removing stop words and punctuations
5.   Stemming - converting words like run running etc to one word
6.  Adding this converted text into a new column
7. Creating word-cloud of spam and not spam

4.	Model building
1.	 For vectorizing text we start with bag of number
2.	 Taking X_trian,y_train
3.	 Taking tfidf vectorizer
4.	 Implementing naïve bayees Multi Dimensional because precision score is 1
5.	 Evaluation
6.	 Improvement
7.	Website implement
1.	 Creating a pipeline
2.	 Connecting with streamlit

