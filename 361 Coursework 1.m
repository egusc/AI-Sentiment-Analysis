%Take data from csv file, split the columns into different tables, and
%tokenize tweets
T = readtable("h-drive/361/361 Coursework 1/text_emotion_data_filtered.csv");
tweets = T(:,2);
labels = T(:,1);
arrtweets = table2array(tweets);
documents = tokenizedDocument(arrtweets);

%Create bag of words from tokenized tweets, remove stop words and
%infrequent words and create tfidf matrix
B = bagOfWords(documents);
newB = removeWords(B,stopWords);
newnewB = removeInfrequentWords(newB, 100);
M = tfidf(newnewB);

%Create matrixes to train the learning algorithm, and matrixes for the
%algorithm to predict labels from
trainingMatrix = full(M(1:6432, 1:84));
trainingLabels = head(labels, 6432);
trainingLabelsArr = table2array(trainingLabels);
predictionMatrix = full(M(6433:8040, 1:84));
predictionLabels = tail(labels, 1608);
predictionLabelsArr = table2array(predictionLabels);

%Create models from training algorithms and predict labels from models
knnmodel = fitcknn(trainingMatrix, trainingLabelsArr);
dtmodel = fitctree(trainingMatrix, trainingLabelsArr);
nbmodel = fitcnb(trainingMatrix, trainingLabelsArr);
predictionsknn = predict(knnmodel, predictionMatrix);
predictionsdt = predict(dtmodel, predictionMatrix);
predictionsnb = predict(nbmodel, predictionMatrix);

%Empty counters for how many labels are correctly guessed
correctLabelknn = 0;
correctLabeldt = 0;
correctLabelnb = 0;

%For each prediction, compare with the actual labels and count number of
%correct predictions
for i = 1:1608
    corr = strcmp(predictionsknn(i), predictionLabelsArr(i));
    if corr == 1
        correctLabelknn = correctLabelknn + 1;
    end
    corr = strcmp(predictionsdt(i), predictionLabelsArr(i));
    if corr == 1
        correctLabeldt = correctLabeldt + 1;
    end
    corr = strcmp(predictionsnb(i), predictionLabelsArr(i));
    if corr == 1
        correctLabelnb = correctLabelnb + 1;
    end
    
    
end

%Calculate and present overall accuracy of predictions for each model
accuracyknn = correctLabelknn / 1608;
accuracydt = correctLabeldt / 1608;
accuracynb = correctLabelnb / 1608;
accuracyknn
accuracydt
accuracynb


