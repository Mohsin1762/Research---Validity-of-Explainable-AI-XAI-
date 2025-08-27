# -*- coding: utf-8 -*-
"""
@author: Syed Muhammad Mohsin
"""
#%% Importing the libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from joypy import joyplot
from datasets import DatasetDict
from datasets import load_dataset
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




#%% Loading the Data
# Attempt to load the dataset with specified split
try:
    ds = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7")
except Exception as e:
    print(f"Error loading dataset: {e}")

dfs = []
# Iterate through each key in the DatasetDict
for key in ds.keys():
    # Get the dataset for the current key
    dataset = ds[key]
    # Convert to DataFrame
    df = dataset.to_pandas()
    # Add a new column for the dataset key
    df['dataset'] = key
    # Append to the list of DataFrames
    dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)
print(combined_df)



combined_dfhead =combined_df.head(100)
dfcounts = combined_df.dataset.value_counts()
anli_df = combined_df[combined_df['dataset'].str.endswith('_ling')]
# Define the top 6 languages based on the earlier reference
top_languages = ['ar_ling', 'es_ling']#, 'zh_ling', 'ur_ling', 'fr_ling', 'ru_ling']
# Select rows corresponding to the top 6 languages
top_languages_df = anli_df[anli_df['dataset'].isin(top_languages)]
print(top_languages_df)



#%% Code for Merging the Labels
# Create a list to hold the rows
data_rows = []

# Loop through the original DataFrame
for index, row in top_languages_df.iterrows():
    # Add human-generated texts (premises)
    data_rows.append({'text': row['premise'], 'label': 0})
    # Add machine-generated texts (hypotheses)
    if row['label'] == 1 or row['label'] == 2:  # Neutral or Contradiction
        data_rows.append({'text': row['hypothesis'], 'label': 1})  # Both are treated as machine-generated

# Create the new DataFrame from the list of rows
new_dataset = pd.DataFrame(data_rows)
print(new_dataset.head())
new_dataset_head = new_dataset.head(100)



#%%% Code for EDA
# Basic information about the dataset
print(new_dataset.info())
# Check for missing values
print(new_dataset.isnull().sum())
# Display basic statistics
print(new_dataset.describe())
##Label Distribution


# Count of each label
label_counts = new_dataset['label'].value_counts()
# Plot the label distribution as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=['Human-Generated (0)', 'Machine-Generated (1)'], 
        autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
plt.title('Label Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()




###Text Length Analysis
# Calculate text lengths
new_dataset['text_length'] = new_dataset['text'].apply(len)
# Prepare data for ridge plot
data_for_ridge = new_dataset[['label', 'text_length']]
data_for_ridge['label'] = data_for_ridge['label'].replace({0: 'Human-Generated (0)', 1: 'Machine-Generated (1)'})
# Create a color map for the two labels
colors = ['lightblue', 'salmon']
# Create the ridge plot without the palette argument
plt.figure(figsize=(10, 6))
joyplot(data=data_for_ridge, by='label', column='text_length', overlap=0.5, fade=True, 
        color=colors)  # Use the defined colors here
plt.title('Ridge Plot of Text Lengths by Label')
plt.xlabel('Text Length')
plt.ylabel('Density')
plt.show()

# Display basic statistics of text lengths
print(new_dataset['text_length'].describe())





### Word Frequency Analysis
# Function to clean and tokenize the text
def tokenize(text):
    # Lowercase and remove punctuation
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens

# Tokenize all texts
all_tokens = []
new_dataset['text'].apply(lambda x: all_tokens.extend(tokenize(x)))
# Count word frequencies
word_counts = Counter(all_tokens)
most_common_words = word_counts.most_common(20)
# Prepare data for plotting
words, counts = zip(*most_common_words)
# Plot the most common words
plt.figure(figsize=(12, 6))
sns.barplot(x=list(words), y=list(counts), palette='mako')
plt.title('Most Common Words')
plt.xticks(rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()




###Sample Texts
# Display sample texts
print("Sample Human-Generated Texts:")
print(new_dataset[new_dataset['label'] == 0]['text'].sample(5, random_state=1).tolist())

print("\nSample Machine-Generated Texts:")
print(new_dataset[new_dataset['label'] == 1]['text'].sample(5, random_state=1).tolist())



### Word Cloud
# Create a word cloud for human-generated texts
human_generated_text = ' '.join(new_dataset[new_dataset['label'] == 0]['text'])
machine_generated_text = ' '.join(new_dataset[new_dataset['label'] == 1]['text'])

# Plotting the word cloud for human-generated texts
plt.figure(figsize=(10, 5))
wordcloud_human = WordCloud(width=800, height=400, background_color='white').generate(human_generated_text)
plt.imshow(wordcloud_human, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Human-Generated Texts')
plt.show()

# Plotting the word cloud for machine-generated texts
plt.figure(figsize=(10, 5))
wordcloud_machine = WordCloud(width=800, height=400, background_color='white').generate(machine_generated_text)
plt.imshow(wordcloud_machine, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Machine-Generated Texts')
plt.show()


######## No of words per language
# Set the aesthetic style of the plots
sns.set(style="whitegrid")
top_languages_df['word_count'] = top_languages_df['hypothesis'].apply(lambda x: len(x.split()))
avg_word_count_per_language = top_languages_df.groupby('dataset')['word_count'].mean().reset_index()
avg_word_count_per_language = avg_word_count_per_language.sort_values(by='word_count', ascending=False)
plt.figure(figsize=(12, 8))
bars = plt.barh(avg_word_count_per_language['dataset'], avg_word_count_per_language['word_count'], 
                 color=sns.color_palette("viridis", len(avg_word_count_per_language)))

# Add titles and labels
plt.title('Average Number of Words per Language', fontsize=16, fontweight='bold')
plt.xlabel('Average Number of Words', fontsize=14)
plt.ylabel('Language', fontsize=14)
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.1f}', va='center', fontsize=12)

# Customize the grid and layout
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



########## Most words for each language
# Function to get the most common words
def get_top_words(data, n=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return sorted_words[:n]

# Languages to analyze
languages = ['ar_ling', 'es_ling']#, 'zh_ling', 'ur_ling', 'fr_ling', 'ru_ling']

# Define colors for each language
colors = ['lightcoral', 'lightgreen', 'lightskyblue', 'lightyellow', 'lightsalmon', 'lightpink']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Loop through each language and plot
for i, lang in enumerate(languages):
    # Filter the dataset for the current language
    lang_data = top_languages_df[top_languages_df['dataset'] == lang]['hypothesis']
    # Get the top 5 words
    top_words = get_top_words(lang_data)
    words, counts = zip(*top_words)  # Unzip the words and their counts
    # Plotting
    bars = axes[i].bar(words, counts, color=colors[i])
    axes[i].set_title(f'Top 5 Words in {lang}', fontsize=14)
    axes[i].set_xlabel('Words', fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(axis='y', linestyle='--')
    # Adding labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

# Adjust layout
plt.tight_layout()
plt.show()



###### TOP BIGRAMS
# Function to get the most common bigrams
def get_top_bigrams(data, n=5):
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    X = vectorizer.fit_transform(data)
    sum_bigrams = X.sum(axis=0)
    bigrams_freq = [(bigram, sum_bigrams[0, idx]) for bigram, idx in vectorizer.vocabulary_.items()]
    sorted_bigrams = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)
    return sorted_bigrams[:n]

# Languages to analyze
languages = ['ar_ling', 'es_ling']#, 'zh_ling', 'ur_ling', 'fr_ling', 'ru_ling']
# Define colors for each language
colors = ['lightcoral', 'lightgreen', 'lightskyblue', 'lightyellow', 'lightsalmon', 'lightpink']
# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Loop through each language and plot
for i, lang in enumerate(languages):
    # Filter the dataset for the current language
    lang_data = top_languages_df[top_languages_df['dataset'] == lang]['hypothesis']
    # Get the top 5 bigrams
    top_bigrams = get_top_bigrams(lang_data)
    bigrams, counts = zip(*top_bigrams)  # Unzip the bigrams and their counts
    # Plotting
    bars = axes[i].bar(bigrams, counts, color=colors[i])
    axes[i].set_title(f'Top 5 Bigrams in {lang}', fontsize=14)
    axes[i].set_xlabel('Bigrams', fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(axis='y', linestyle='--')
    # Adding labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

# Adjust layout
plt.tight_layout()
plt.show()

#%% Extensive EDA

nltk.download('punkt')
# Assuming top_languages_df contains the text data with a 'hypothesis' column and 'dataset' for languages
top_languages_df['sentence_count'] = top_languages_df['hypothesis'].apply(lambda x: len(sent_tokenize(x)))
top_languages_df['word_count'] = top_languages_df['hypothesis'].apply(lambda x: len(x.split()))
top_languages_df['avg_sentence_length'] = top_languages_df['word_count'] / top_languages_df['sentence_count']

# Group by language (dataset) and calculate the average sentence length
avg_sentence_length_per_language = top_languages_df.groupby('dataset')['avg_sentence_length'].mean().reset_index()
avg_sentence_length_per_language = avg_sentence_length_per_language.sort_values(by='avg_sentence_length', ascending=False)

# Plotting the average sentence length per language as a horizontal bar graph
plt.figure(figsize=(10, 8))
bars = plt.barh(avg_sentence_length_per_language['dataset'], avg_sentence_length_per_language['avg_sentence_length'], color='skyblue')
plt.title('Average Sentence Length per Language', fontsize=16, fontweight='bold')
plt.xlabel('Average Sentence Length (words)', fontsize=14)
plt.ylabel('Language', fontsize=14)

# Adding labels on bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}', va='center', fontsize=12)

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




## Similarity Code
# Function to get sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis
new_dataset['sentiment'] = new_dataset['text'].apply(get_sentiment)
# Calculate average sentiment by label
avg_sentiment = new_dataset.groupby('label')['sentiment'].mean().reset_index()
# Set the style for the plot
sns.set(style="whitegrid")
# Plotting the average sentiment by label
plt.figure(figsize=(8, 6))
bar_plot = sns.barplot(x='label', y='sentiment', data=avg_sentiment, palette='pastel')
# Add data labels on top of the bars
for p in bar_plot.patches:
    bar_plot.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom', fontsize=12)

# Set titles and labels
plt.title('Average Sentiment by Text Type', fontsize=18, fontweight='bold')
plt.xlabel('Label', fontsize=14)
plt.ylabel('Average Sentiment', fontsize=14)
plt.xticks([0, 1], ['Human-Generated (0)', 'Machine-Generated (1)'])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Customize layout
plt.tight_layout()
plt.show()




#%% Similarity Analysis and Plotting

# Select a smaller sample size for human and machine-generated texts
sample_size = 1000  # Adjust this based on memory availability

# Sample human-generated and machine-generated texts
human_texts_sample = new_dataset[new_dataset['label'] == 0].sample(n=sample_size, random_state=42)['text']
machine_texts_sample = new_dataset[new_dataset['label'] == 1].sample(n=sample_size, random_state=42)['text']

# Combine the sampled texts for vectorization
combined_sample_texts = pd.concat([human_texts_sample, machine_texts_sample])

# Fit and transform the TF-IDF vectorizer on the combined sample texts
tfidf_sample_matrix = tfidf_vectorizer.fit_transform(combined_sample_texts)

# Split the TF-IDF matrix into human and machine-generated text based on labels
human_sample_tfidf = tfidf_sample_matrix[:sample_size]
machine_sample_tfidf = tfidf_sample_matrix[sample_size:]

# Compute cosine similarities between human and machine-generated texts
similarity_human_machine_sample = cosine_similarity(human_sample_tfidf, machine_sample_tfidf)

# Flatten the array for plotting
human_machine_sim_flat_sample = similarity_human_machine_sample.flatten()

# Plotting the similarity distribution between human and machine-generated texts
plt.figure(figsize=(12, 6))

# Use 'stat="probability"' to normalize the y-axis to probability values
sns.histplot(human_machine_sim_flat_sample, color='green', label='Human-Machine Similarity', kde=True, bins=30, stat="probability")

# Add labels and title
plt.title('Cosine Similarity Distribution between Human and Machine-Generated Texts', fontsize=16)
plt.xlabel('Cosine Similarity', fontsize=14)
plt.ylabel('Probability', fontsize=14)  # Change label to Probability
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()







#%% Code for Model
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
# Step 1: Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokenize the text
encodings = tokenizer(new_dataset['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
# Step 2: Split the Data
# Instead of splitting just input_ids, keep all encodings in a dictionary
train_size = int(0.8 * len(encodings['input_ids']))  # 80% for training
X_train = {key: val[:train_size] for key, val in encodings.items()}  # Training data
X_test = {key: val[train_size:] for key, val in encodings.items()}    # Testing data

# Corresponding labels
y_train, y_test = train_test_split(new_dataset['label'], test_size=0.2)

# Step 3: Prepare the Dataset for PyTorch
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}  # Access encodings as a dict
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Prepare the datasets
train_dataset = TextDataset(encodings=X_train, labels=y_train.tolist())
test_dataset = TextDataset(encodings=X_test, labels=y_test.tolist())

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}



# Step 4: Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Step 5: Set up the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Update the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Add this line
)

# Step 6: Train the model
trainer.train()


eval_results = trainer.evaluate()
print(eval_results)


# Print evaluation results
accuracy = eval_results['eval_accuracy']
print(f'Accuracy: {accuracy * 100:.2f}%')


#%% CODE FOR MODEL


#%% Feature Extraction with TF-IDF
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Fit and transform the text data
X = tfidf_vectorizer.fit_transform(new_dataset['text'])

# Labels
y = new_dataset['label']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training and testing data split completed.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

#%% Training a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Define the logistic regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print out the performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




#%% XAI

#%% SHAP Analysis
import shap
# Create a SHAP explainer using the trained model
explainer = shap.LinearExplainer(logistic_model, X_train, feature_perturbation='interventional')
# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)
# Visualize SHAP values for a random sample of 10 texts
shap.summary_plot(shap_values, X_test, feature_names=tfidf_vectorizer.get_feature_names_out(), max_display=10)


#%% Training an XGBoost Model
from xgboost import XGBClassifier
# Define the XGBoost model
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
# Train the model
xgb_model.fit(X_train, y_train)
# Make predictions
y_pred_xgb = xgb_model.predict(X_test)
# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")
print(f"XGBoost Precision: {precision_xgb:.4f}")
print(f"XGBoost Recall: {recall_xgb:.4f}")
print(f"XGBoost F1 Score: {f1_xgb:.4f}")



#%% SHAP Analysis for XGBoost Model
import shap
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test)

# Plot the SHAP summary plot
shap.summary_plot(shap_values_xgb, X_test, plot_type="bar")
shap.summary_plot(shap_values_xgb, X_test)



#%%
# Visualize a single instance from the test set
shap.initjs()  # Initialize JavaScript visualization
instance_index = 0  # Choose the instance to explain

# Convert the sparse matrix to a dense array for SHAP
X_test_dense = X_test.toarray()

# Generate the force plot for the specified instance
shap.force_plot(
    explainer.expected_value, 
    shap_values[instance_index], 
    X_test_dense[instance_index], 
    feature_names=tfidf_vectorizer.get_feature_names_out()
)
    



#%% Final code


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap

#%% Step 1: Vectorization Techniques
# Define vectorizers
vectorizers = {
    'tfidf': TfidfVectorizer(max_features=1000),
    'count': CountVectorizer(max_features=1000)
}

# Dictionary to store results
results = {}

# Loop through each vectorization method
for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #%% Step 2: Train XGBoost Model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store metrics
    results[vec_name] = {
        'accuracy': accuracy,
        'classification_report': report
    }

    #%% Step 3: XAI with SHAP
    # Convert sparse matrix to dense
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Initialize the SHAP explainer with dense data
    explainer = shap.Explainer(model, X_train_dense)
    shap_values = explainer(X_test_dense)
    

    # Save SHAP summary plot (optional)
    shap.summary_plot(shap_values, X_test, feature_names=vectorizer.get_feature_names_out(), show=False)
    plt.savefig(f"{vec_name}_shap_summary.png")

#%% Step 4: Create Summary Table
summary_df = pd.DataFrame.from_dict({vec: metrics['accuracy'] for vec, metrics in results.items()}, orient='index', columns=['Accuracy'])
summary_df['Precision'] = [results[vec]['classification_report']['weighted avg']['precision'] for vec in results]
summary_df['Recall'] = [results[vec]['classification_report']['weighted avg']['recall'] for vec in results]
summary_df['F1 Score'] = [results[vec]['classification_report']['weighted avg']['f1-score'] for vec in results]

print(summary_df)







----------------- Proving Validity

1. Conceptualizing Validity in XAI:
    
Faithfulness: This refers to how accurately the explanations reflect the model's inner workings. For example, if SHAP assigns high importance to certain features, these features should genuinely influence the model's predictions.
Consistency: An explanation should remain consistent when applied to similar instances. For example, the SHAP values should not vary drastically when slight changes are made to similar inputs.
Human Interpretability: Consider whether the explanations generated (e.g., SHAP values) are understandable to end-users, especially when dealing with multilingual text.


import numpy as np
from scipy.stats import pearsonr

# Dictionary to store correlation results
correlations = {}

# Loop through each vectorization method
for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)

    # Calculate feature importance from the model
    model_feature_importance = model.feature_importances_

    # Convert sparse matrix to dense for SHAP
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Initialize SHAP Explainer and compute SHAP values
    explainer = shap.Explainer(model, X_train_dense)
    shap_values = explainer(X_test_dense)

    # Calculate mean absolute SHAP values for each feature
    shap_feature_importance = np.abs(shap_values.values).mean(axis=0)

    # Calculate Pearson correlation between model and SHAP feature importances
    correlation, _ = pearsonr(model_feature_importance, shap_feature_importance)
    correlations[vec_name] = correlation
    print(f"Correlation between SHAP and model feature importance for {vec_name}: {correlation}")



--------------------
Step 2: Ablation Tests to Validate SHAP Explanations
Ablation tests involve the following:

Identify Important Features using SHAP values.
Create Reduced Datasets by selecting only the top features (e.g., top 10 or 20 based on SHAP importance) and observe how the model performs with this reduced feature set.
Evaluate Model Performance on the reduced dataset and compare it to the original performance.


Interpretation of the Results:
Performance Drop Magnitude: A performance drop of around 0.07-0.10 means that the removed features
do impact the model, but the drop is not extremely high. This implies that the selected features are
important but not solely responsible for the model's predictive power.
 
Comparison of Vectorization Techniques: The greater drop for Count vectorization could mean that the 
important features identified by SHAP are more concentrated for this method compared to TF-IDF,
where the influence of important features might be spread more evenly across a larger set.
 
These findings can be used to argue about the validity of SHAP-based explanations—showing that 
SHAP does identify influential features, though the overall impact on model performance when 
using a reduced set of features is moderate.




# Dictionary to store ablation test results
reduced_accuracies = {}

# Define the number of top features to retain
top_n_features = 20

for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the original XGBoost Model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate the original model
    y_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, y_pred)

    # Convert sparse matrix to dense for SHAP
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Compute SHAP values
    explainer = shap.Explainer(model, X_train_dense)
    shap_values = explainer(X_test_dense)

    # Calculate mean absolute SHAP values and get top features
    shap_feature_importance = np.abs(shap_values.values).mean(axis=0)
    top_features = np.argsort(shap_feature_importance)[-top_n_features:]

    # Create a reduced training and test set with only the top features
    X_train_reduced = X_train_dense[:, top_features]
    X_test_reduced = X_test_dense[:, top_features]

    # Train a new XGBoost model on the reduced feature set
    reduced_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    reduced_model.fit(X_train_reduced, y_train)

    # Evaluate the reduced model
    y_pred_reduced = reduced_model.predict(X_test_reduced)
    reduced_accuracy = accuracy_score(y_test, y_pred_reduced)
    reduced_accuracies[vec_name] = reduced_accuracy

    # Calculate and print the drop in accuracy
    performance_drop = original_accuracy - reduced_accuracy
    print(f"Performance drop for {vec_name} after using only top {top_n_features} features: {performance_drop}")

# Add ablation performance drop to summary table
summary_df['Ablation Performance Drop'] = [original_accuracy - reduced_accuracies[vec] for vec in results]
print(summary_df)




-------------------------
import numpy as np

# Define the perturbation level (e.g., 5% of data)
perturbation_fraction = 0.05

# Dictionary to store sensitivity results
sensitivity_results = {}

for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Convert the dense test array to float64 for perturbation
    perturbed_X_test = X_test_dense.astype(np.float64)
    
    # Train the original model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)
    
    # Compute SHAP values for the original data
    explainer = shap.Explainer(model, X_train_dense)
    original_shap_values = explainer(X_test_dense).values
    
    # Perturb a small fraction of the data
    perturbation_size = int(perturbation_fraction * X_test_dense.shape[0])
    perturbed_indices = np.random.choice(X_test_dense.shape[0], perturbation_size, replace=False)
    
    # Add noise to the selected rows
    noise = np.random.normal(loc=0, scale=0.01, size=perturbed_X_test[perturbed_indices].shape)
    perturbed_X_test[perturbed_indices] += noise
    
    # Compute SHAP values for the perturbed data
    perturbed_shap_values = explainer(perturbed_X_test).values
    
    # Calculate the sensitivity as the mean absolute difference between original and perturbed SHAP values
    shap_sensitivity = np.mean(np.abs(original_shap_values - perturbed_shap_values))
    sensitivity_results[vec_name] = shap_sensitivity
    
    print(f"Sensitivity of SHAP values for {vec_name}: {shap_sensitivity}")

# Print out the sensitivity analysis results
print("Sensitivity Analysis Results:", sensitivity_results)











----------------- LIME


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np

#%% Step 1: Vectorization Techniques
# Define vectorizers
vectorizers = {
    'tfidf': TfidfVectorizer(max_features=1000),
    'count': CountVectorizer(max_features=1000)
}

# Dictionary to store results
results = {}

# Loop through each vectorization method
for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #%% Step 2: Train XGBoost Model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store metrics
    results[vec_name] = {
        'accuracy': accuracy,
        'classification_report': report
    }

    #%% Step 3: XAI with SHAP
    # Convert sparse matrix to dense
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Initialize the SHAP explainer with dense data
    explainer = shap.Explainer(model, X_train_dense)
    shap_values = explainer(X_test_dense)
    
    # Save SHAP summary plot (optional)
    shap.summary_plot(shap_values, X_test_dense, feature_names=vectorizer.get_feature_names_out(), show=False)
    plt.savefig(f"{vec_name}_shap_summary.png")

    #%% Step 4: XAI with LIME
#%% Step 4: XAI with LIME
    lime_explainer = LimeTextExplainer(class_names=["Human-generated", "Machine-generated"])
    
    # Get a few test samples to explain
    num_samples_to_explain = 5
    for i in range(num_samples_to_explain):
        # Choose a sample from the test set
        text_instance = X_test[i]  # Get the i-th test sample
        text_instance_str = new_dataset['text'][X_test.indices[i]]  # Get the original text from the dataset
        
        # Explain the prediction using the vectorized instance
        exp = lime_explainer.explain_instance(
            text_instance_str,
            lambda x: model.predict_proba(vectorizer.transform(x)),
            num_features=10
        )
        
        # Display the explanation
        print(f"LIME Explanation for instance {i}:")
        print(exp.as_list())
        
        # Save LIME visualization (optional)
        exp.save_to_file(f"{vec_name}_lime_explanation_{i}.html")


#%% Step 5: Create Summary Table
summary_df = pd.DataFrame.from_dict({vec: metrics['accuracy'] for vec, metrics in results.items()}, orient='index', columns=['Accuracy'])
summary_df['Precision'] = [results[vec]['classification_report']['weighted avg']['precision'] for vec in results]
summary_df['Recall'] = [results[vec]['classification_report']['weighted avg']['recall'] for vec in results]
summary_df['F1 Score'] = [results[vec]['classification_report']['weighted avg']['f1-score'] for vec in results]

print(summary_df)






-----------  VALIDITY OF XAI


from sklearn.metrics import accuracy_score, classification_report
from lime.lime_text import LimeTextExplainer
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Dictionary to store correlation results
lime_correlations = {}

# Loop through each vectorization method
for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost Model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate feature importance from the model
    model_feature_importance = model.feature_importances_
    
    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=['human', 'machine'])
    
    # Store LIME feature importances for correlation calculation
    lime_feature_importances = []
    
    # Get LIME explanations for a few instances in the test set
    num_samples_to_explain = 5  # Adjust as necessary
    for i in range(num_samples_to_explain):
        # Get the original text for the i-th test instance
        text_instance = new_dataset['text'].iloc[X_test.indices[i]]
        
        # Explain the prediction using the vectorized instance
        exp = explainer.explain_instance(
            text_instance,
            lambda x: model.predict_proba(vectorizer.transform(x)),
            num_features=10  # Use a consistent number of features
        )
        
        # Get LIME feature importances, ensuring consistent shape
        lime_feature_importance = np.array([explanation[1] for explanation in exp.as_list()])
        lime_feature_importances.append(lime_feature_importance)
    
    # Pad or truncate LIME feature importances to ensure they have the same length
    max_len = max(len(x) for x in lime_feature_importances)
    padded_lime_importances = [np.pad(x, (0, max_len - len(x)), mode='constant') for x in lime_feature_importances]
    
    # Calculate average LIME feature importances for correlation
    lime_feature_importances_avg = np.mean(padded_lime_importances, axis=0)

    # Check if model feature importance length matches LIME feature importances
    if len(model_feature_importance) == len(lime_feature_importances_avg):
        # Calculate Pearson correlation between model and LIME feature importances
        correlation, _ = pearsonr(model_feature_importance, lime_feature_importances_avg)
        lime_correlations[vec_name] = correlation
        print(f"Correlation between LIME and model feature importance for {vec_name}: {correlation}")
    else:
        print(f"Length mismatch for {vec_name}: Model features: {len(model_feature_importance)}, LIME features: {len(lime_feature_importances_avg)}")




----- ABLATION TESTS


from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

lime_reduced_accuracies = {}

# Define the number of top features to retain
top_n_features = 20

for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the original XGBoost Model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate the original model
    y_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, y_pred)
    
    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=['human', 'machine'])
    
    # Get LIME explanations for individual instances in the test set
    lime_feature_importances_list = []
    for i in range(X_test.shape[0]):  # Iterate through each instance in X_test
        text_instance = new_dataset['text'].iloc[X_test.indices[i]]  # Get the original text
        exp = explainer.explain_instance(
            text_instance,
            lambda x: model.predict_proba(vectorizer.transform(x)),
            num_features=1000
        )
        
        # Get LIME feature importances
        lime_feature_importances = np.array([explanation[1] for explanation in exp.as_list()])
        lime_feature_importances_list.append(lime_feature_importances)
    
    # Calculate average LIME feature importances
    lime_feature_importances_avg = np.mean(lime_feature_importances_list, axis=0)

    # Get the top features
    top_features_indices = np.argsort(lime_feature_importances_avg)[-top_n_features:]
    
    # Create a reduced training and test set with only the top features
    X_train_reduced = X_train[:, top_features_indices]
    X_test_reduced = X_test[:, top_features_indices]
    
    # Train a new XGBoost model on the reduced feature set
    reduced_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    reduced_model.fit(X_train_reduced, y_train)
    
    # Evaluate the reduced model
    y_pred_reduced = reduced_model.predict(X_test_reduced)
    reduced_accuracy = accuracy_score(y_test, y_pred_reduced)
    lime_reduced_accuracies[vec_name] = reduced_accuracy
    
    # Calculate and print the drop in accuracy
    performance_drop = original_accuracy - reduced_accuracy
    print(f"Performance drop for {vec_name} after using only top {top_n_features} features: {performance_drop}")



-------------- SENSITIVITY  LIME

# Define the perturbation level (e.g., 5% of data)
perturbation_fraction = 0.05

# Dictionary to store sensitivity results
lime_sensitivity_results = {}

for vec_name, vectorizer in vectorizers.items():
    # Apply vectorization
    X = vectorizer.fit_transform(new_dataset['text'])
    y = new_dataset['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the original model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=13, random_state=42)
    model.fit(X_train, y_train)

    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=['human', 'machine'])

    # Get LIME explanations for the original test set
    lime_explanations = explainer.explain_instance(X_test, model.predict_proba, num_features=1000)
    
    # Store the original feature importances
    original_lime_importances = np.array([explanation[1] for explanation in lime_explanations.as_list()])

    # Perturb a small fraction of the data
    perturbation_size = int(perturbation_fraction * X_test.shape[0])
    perturbed_indices = np.random.choice(X_test.shape[0], perturbation_size, replace=False)

    # Perturb the selected rows
    perturbed_X_test = X_test.copy()
    noise = np.random.normal(loc=0, scale=0.01, size=perturbed_X_test[perturbed_indices].shape)
    perturbed_X_test[perturbed_indices] += noise
    
    # Get LIME explanations for the perturbed data
    perturbed_lime_explanations = explainer.explain_instance(perturbed_X_test, model.predict_proba, num_features=1000)
    
    # Store the perturbed feature importances
    perturbed_lime_importances = np.array([explanation[1] for explanation in perturbed_lime_explanations.as_list()])
    
    # Calculate the sensitivity as the mean absolute difference between original and perturbed LIME importances
    lime_sensitivity = np.mean(np.abs(original_lime_importances - perturbed_lime_importances))
    lime_sensitivity_results[vec_name] = lime_sensitivity
    
    print(f"Sensitivity of LIME values for {vec_name}: {lime_sensitivity}")

# Print out the sensitivity analysis results
print("LIME Sensitivity Analysis Results:", lime_sensitivity_results)
