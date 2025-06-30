# Decision-Trees-and-Random-Forests
Decision Trees and Random Forests
In this task, we aimed to build and compare tree-based models for classification using the Heart Disease Dataset. The objective was to predict the presence or absence of heart disease based on patient features using Decision Tree and Random Forest models. We used tools like Pandas, Scikit-learn, Matplotlib, and Seaborn to accomplish this.

We began by importing and loading the dataset using Pandas. We checked the shape and data types of each column to understand the dataset structure. Since the dataset was already encoded (i.e., all categorical columns were converted into numeric), no manual encoding was necessary.

Next, we split the dataset into features (X) and target variable (y). The target variable indicated whether heart disease was present or not (1 or 0). We then split this data into training and testing sets using train_test_split from Scikit-learn, keeping 80% for training and 20% for testing. To ensure model performance was not affected by feature scale, we applied StandardScaler from Scikit-learn to scale only the feature values (X), not the target (y).

We then trained a Decision Tree Classifier using DecisionTreeClassifier() from Scikit-learn. After training, we evaluated the model using accuracy score and classification report, which includes precision, recall, and F1-score. We also visualized the trained decision tree using plot_tree() from Matplotlib, which showed how the model splits the data at each node.

To improve accuracy and reduce overfitting, we implemented a Random Forest Classifier, an ensemble model using multiple decision trees. We trained it using RandomForestClassifier(n_estimators=100) from Scikit-learn, where 100 trees were created. This model gave perfect accuracy on test data.

To understand which features were most important in making predictions, we accessed the feature importances from the random forest model using feature_importances_. We stored and sorted these importances using Pandas, which helped identify which health attributes contributed most to the model’s prediction.

Finally, we performed cross-validation using cross_val_score from Scikit-learn with 5 folds. This helped ensure that our model was not overfitting and could generalize well to new data. The average accuracy across the folds was very high, showing that our model was robust.
