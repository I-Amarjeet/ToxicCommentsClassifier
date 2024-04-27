val_border = int(len(train_balanced)*0.6)
validation_set = train_balanced[val_border:].copy()

vectorizer = TfidfVectorizer(ngram_range=(1, 5), max_features=5000)
vectorizer.fit(train_balanced['comment_text'])
vectorizer

x_train = vectorizer.transform(train_balanced['comment_text'])

lr_classifier = LogisticRegression(solver='liblinear')
average_roc = 0
    
for label in target_columns:
    lr_classifier.fit(x_train[:val_border], train_balanced[label][:val_border])
    predictions = lr_classifier.predict(x_train[val_border:])
    print(f'Label = {label}')
    print(classification_report(validation_set[label], predictions))
    print(f'AUC: {roc_auc_score(validation_set[label], predictions)}')
    average_roc += roc_auc_score(validation_set[label], predictions)
                                 
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')