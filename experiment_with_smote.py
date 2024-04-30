from imblearn.over_sampling import SMOTE

# List of imbalanced categories
imbalanced_categories = ['severe_toxic', 'identity_hate', 'threat']

# Apply SMOTE to each imbalanced category separately
for category in imbalanced_categories:
    smote = SMOTE(random_state=42)
    X_imbalanced = df_balanced['comment_text'].values
    y_imbalanced = df_balanced[category].values
    X_resampled, y_resampled = smote.fit_resample(X_imbalanced, y_imbalanced)
    
    #append the resampled data to the balanced dataframe
    df_resampled = pd.DataFrame(data={'comment_text': X_resampled, category: y_resampled})
   
    
    # Concatenate with the balanced toxic and non-toxic categories
    df_balanced = pd.concat([df_balanced, df_resampled])

# Shuffle the DataFrame
df_balanced = df_balanced.sample(frac=1)

# Plot the distribution after applying SMOTE
toxicity_counts_smote = df_balanced.iloc[:, 2:].apply(pd.Series.value_counts)
category_totals_smote = toxicity_counts_smote.iloc[1].sort_values(ascending=False)

plt.figure(figsize=(20, 10))
bp = sns.barplot(x=category_totals_smote.index, y=category_totals_smote.values)
for i in range(len(category_totals_smote.values)):
    bp.text(i, category_totals_smote.values[i], category_totals_smote.values[i], ha = 'center')
bp.set_yscale("log")
bp.tick_params(labelsize=15)



