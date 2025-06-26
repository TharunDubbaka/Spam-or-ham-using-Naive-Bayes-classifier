# Spam-or-ham-using-Naive-Bayes-classifier
#this is the notes for my project.
#The feature max value is being tested manually
#i am going to start with 3000 and probably go upto 7000
#There is a risk of overfitting
#I will just compare the accuracies of each max feature value and get to the best one
#the encoding is done using LabelEncoder"
#print(y[:10]) to check the first encoded values
"""Classification report for training data set :  
      precision    recall  f1-score   support

           0       0.98      1.00      0.99      3867
           1       1.00      0.87      0.93       590
           2       0.00      0.00      0.00         1

    accuracy                           0.98      4458
   macro avg       0.66      0.62      0.64      4458
weighted avg       0.98      0.98      0.98      4458

Classification report for testing data set :                precision    recall  f1-score   support

           0       0.98      1.00      0.99       958
           1       0.99      0.85      0.91       157

    accuracy                           0.98      1115
   macro avg       0.98      0.92      0.95      1115
weighted avg       0.98      0.98      0.98      1115"""
# I got an extra "2" in the precision that means there is a missing label
#['ham' 'spam' '{"mode":"full"'] the mode full one is the problem could be fixed easily
#project is done
#i tried to increase the feature max to 4000
#but the output was same so it was useless and may lead to overfitting so i just use 3000
#the classificarion report is
"""Classification report for training data set :                precision    recall  f1-score   support

           0       0.98      1.00      0.99      3859
           1       1.00      0.87      0.93       598

    accuracy                           0.98      4457
   macro avg       0.99      0.94      0.96      4457
weighted avg       0.98      0.98      0.98      4457

Classification report for testing data set :                precision    recall  f1-score   support

           0       0.98      1.00      0.99       966
           1       1.00      0.85      0.92       149

    accuracy                           0.98      1115
   macro avg       0.99      0.93      0.95      1115
weighted avg       0.98      0.98      0.98      1115"""
# it shows that my model is 98% accurate in both training and testing sets
#Nice the model worked for real world sentence too phew....
#I am done, i will deploy later
#My model achieved high accuracy, but after inspecting the confusion matrix, I realized it was missing some spam messages, so I fine-tuned the threshold and improved recall.
