AutoML<-function(DataFrame,Split_Value,Size,SMOTE){

  #======================================
  #Library to use Weka functions
  library(RWeka)
  #Libraries needed to normalize the numeric values
  library(cluster)
  library(MASS)
  library(clusterSim)
  #Library to use SMOTE to balance the classes
  library(grid)
  library(DMwR) 
  #Library to split values
  library(caTools)
  
  #============Resampling data ================
  
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  
  DataFrame<-resample(target~ .,data=DataFrame,control=Weka_control(Z=Size))
  
  #===========Numeric Columns to be normalized
  columns_to_change<-colnames(select_if(DataFrame, is.numeric))
  #Normalize Dataframe
  for(i in 1:length(columns_to_change)){column_number<-which(colnames(DataFrame)==columns_to_change[i])  
  DataFrame[,column_number]<-data.Normalization (DataFrame[,column_number] ,type="n1",normalization="column")
  }
  
  
  #===========Split File
  data=DataFrame
  split = sample.split(data$target, SplitRatio = Split_Value)
  training_set = subset(data, split == TRUE)
  test_set = subset(data, split == FALSE)
  
  
  #===========Applying SMOTE
    
  if (SMOTE=='Y') {
    #str(training_set$target)
    prop.table(table(training_set$target))
    training_set<-SMOTE(target ~ ., training_set, perc.over = 100, perc.under=200)
    #prop.table(table(training_set$target))
    
  }
    
  #Load file
  train_file_clean <- training_set  
  #Check the file summary
  #summary(train_file)
  
  
  #============Setting up variables for further calculations===================
  #Count total records
  Total_Records<-sqldf("Select count(*) from train_file_clean")
  Total_Records<-Total_Records$`count(*)`
  # Check sensibity and 
  summary(train_file_clean$target)/(Total_Records)
  
  #write.table(train_file_clean, file = "~/train_file_clean.csv", sep = ",", col.names = NA, qmethod = "double")   
  
  
  #===============Building the Models===================
  #================Classification==============#
  #BayesNet
  #naiveBayes
  #Logistic
  #MultilayerPerceptron
  #SMO
  #Bagging
  #LogitBoost
  #DecistionTable
  #OneR
  #Part
  #ZeroR
  #DesicionStump #---- RUN cv
  #J48
  #LMT
  #randomForest         ----------Java.lang.OutOfMemoryError
  #Randomtree
  #REPTree
  
  #===============ZeroR===================
  ZeroR<-make_Weka_classifier("weka/classifiers/rules/ZeroR")
  ZeroR_Classifier<-ZeroR(train_file_clean$target~ ., data = train_file_clean)
  ZeroR_Train<-summary(ZeroR_Classifier)
  #Cross Validation
  ZeroR_CV <- evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  ZeroR_Test<-table( predict(ZeroR_Classifier,newdata=test_set),test_set$target )
  
  #===============OneR===================
  OneR_Classifier<-OneR(train_file_clean$target~ ., data = train_file_clean)
  OneR_Train<-summary(OneR_Classifier)
  #Cross Validation
  OneR_CV <- evaluate_Weka_classifier(OneR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  OneR_Test<-table( predict(OneR_Classifier,newdata=test_set),test_set$target )
  #===============MultiLayerPerceptron===================
  MultilayerPerceptron<-make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron")
  MultilayerPerceptron_Classifier<-MultilayerPerceptron(train_file_clean$target~ ., data = train_file_clean)
  MultilayerPerceptron_Train<-summary(MultilayerPerceptron_Classifier)
  #Cross Validation
  MultilayerPerceptron_CV <- evaluate_Weka_classifier(MultilayerPerceptron_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  MultilayerPerceptron_Test<-table( predict(MultilayerPerceptron_Classifier,newdata=test_set),test_set$target )
  if(!exists("MultilayerPerceptron_Test")){MultilayerPerceptron_Test<-summary(ZeroR_Classifier)}
  if(!exists("MultilayerPerceptron_Train")){MultilayerPerceptron_Train<-summary(ZeroR_Classifier)}
  if(!exists("MultilayerPerceptron_CV")){MultilayerPerceptron_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============J48===================
  J48_Classifier<-J48(train_file_clean$target~ ., data = train_file_clean)
  J48_Train<-summary(J48_Classifier)
  #Cross Validation
  J48_CV <- evaluate_Weka_classifier(J48_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  J48_Test<-table( predict(J48_Classifier,newdata=test_set),test_set$target )
  if(!exists("J48_Test")){J48_Test<-summary(ZeroR_Classifier)}
  if(!exists("J48_Train")){J48_Train<-summary(ZeroR_Classifier)}
  if(!exists("J48_CV")){J48_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============IBk===================
  IBk_Classifier<-IBk(train_file_clean$target~ ., data = train_file_clean,control=Weka_control(K=1))
  IBK_Train<-summary(IBk_Classifier)
  #Cross Validation
  IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  IBk_Test<-table( predict(IBk_Classifier,newdata=test_set),test_set$target )
  if(!exists("IBk_Test")){IBk_Test<-summary(ZeroR_Classifier)}
  if(!exists("IBK_Train")){IBK_Train<-summary(ZeroR_Classifier)}
  if(!exists("IBk_CV")){IBk_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============BayesNet===================
  BayesNet<-make_Weka_classifier("weka/classifiers/bayes/BayesNet")
  BayesNet_Classifier<-BayesNet(train_file_clean$target~ ., data = train_file_clean)
  BayesNet_Train<-summary(BayesNet_Classifier)
  #Cross Validation
  BayesNet_CV <- evaluate_Weka_classifier(BayesNet_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  BayesNet_Test<-table( predict(BayesNet_Classifier,newdata=test_set),test_set$target )
  if(!exists("BayesNet_Test")){BayesNet_Test<-summary(ZeroR_Classifier)}
  if(!exists("BayesNet_Train")){BayesNet_Train<-summary(ZeroR_Classifier)}
  if(!exists("BayesNet_CV")){BayesNet_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============NaiveBayes===================
  NaiveBayes<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
  NaiveBayes_Classifier<-NaiveBayes(train_file_clean$target~ ., data = train_file_clean)
  NaiveBayes_Train<-summary(NaiveBayes_Classifier)
  #Cross Validation
  NaiveBayes_CV <- evaluate_Weka_classifier(NaiveBayes_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  NaiveBayes_Test<-table( predict(NaiveBayes_Classifier,newdata=test_set),test_set$target )
  if(!exists("NaiveBayes_Test")){NaiveBayes_Test<-summary(ZeroR_Classifier)}
  if(!exists("NaiveBayes_Train")){NaiveBayes_Train<-summary(ZeroR_Classifier)}
  if(!exists("NaiveBayes_CV")){NaiveBayes_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============Logistic===================
  Logistic_Classifier<-Logistic(train_file_clean$target~ ., data = train_file_clean)
  Logistic_Train<-summary(Logistic_Classifier)
  #Cross Validation
  Logistic_CV <- evaluate_Weka_classifier(Logistic_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  Logistic_Test<-table( predict(Logistic_Classifier,newdata=test_set),test_set$target )
  if(!exists("Logistic_Test")){Logistic_Test<-summary(ZeroR_Classifier)}
  if(!exists("Logistic_Train")){Logistic_Train<-summary(ZeroR_Classifier)}
  if(!exists("Logistic_CV")){Logistic_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============SMO===================
  SMO_Classifier<-SMO(train_file_clean$target~ ., data = train_file_clean)
  SMO_Train<-summary(SMO_Classifier)
  #Cross Validation
  SMO_CV <- evaluate_Weka_classifier(SMO_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  SMO_Test<-table( predict(SMO_Classifier,newdata=test_set),test_set$target )
  if(!exists("SMO_Test")){SMO_Test<-summary(ZeroR_Classifier)}
  if(!exists("SMO_Train")){SMO_Train<-summary(ZeroR_Classifier)}
  if(!exists("SMO_CV")){SMO_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============LMT===================
  LMT_Classifier<-LMT(train_file_clean$target~ ., data = train_file_clean, na.action=NULL)
  LMT_Train<-summary(LMT_Classifier)
  #Cross Validation
  LMT_CV <- evaluate_Weka_classifier(LMT_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  LMT_Test<-table( predict(LMT_Classifier,newdata=test_set),test_set$target )
  if(!exists("LMT_Test")){LMT_Test<-summary(ZeroR_Classifier)}
  if(!exists("LMT_Train")){LMT_Train<-summary(ZeroR_Classifier)}
  if(!exists("LMT_CV")){LMT_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============RandomForest===================
  RandomForest<-make_Weka_classifier("weka/classifiers/trees/RandomForest")
  RandomForest_Classifier<-RandomForest(train_file_clean$target~ ., data = train_file_clean)
  RandomForest_Train<-summary(RandomForest_Classifier)
  #Cross Validation
  RandomForest_CV <- evaluate_Weka_classifier(RandomForest_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  RandomForest_Test<-table( predict(RandomForest_Classifier,newdata=test_set),test_set$target )
  if(!exists("RandomForest_Test")){RandomForest_Test<-summary(ZeroR_Classifier)}
  if(!exists("RandomForest_Train")){RandomForest_Train<-summary(ZeroR_Classifier)}
  if(!exists("RandomForest_CV")){RandomForest_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============RandomTree===================
  RandomTree<-make_Weka_classifier("weka/classifiers/trees/RandomTree")
  RandomTree_Classifier<-RandomTree(train_file_clean$target~ ., data = train_file_clean)
  RandomTree_Train<-summary(RandomTree_Classifier)
  #Cross Validation
  RandomTree_CV <- evaluate_Weka_classifier(RandomTree_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  RandomTree_Test<-table( predict(RandomTree_Classifier,newdata=test_set),test_set$target )
  if(!exists("RandomTree_Test")){RandomTree_Test<-summary(ZeroR_Classifier)}
  if(!exists("RandomTree_Train")){RandomTree_Train<-summary(ZeroR_Classifier)}
  if(!exists("RandomTree_CV")){RandomTree_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============REPTree===================
  REPTree<-make_Weka_classifier("weka/classifiers/trees/REPTree")
  REPTree_Classifier<-REPTree(train_file_clean$target~ ., data = train_file_clean)
  REPTree_Train<-summary(REPTree_Classifier)
  #Cross Validation
  REPTree_CV <- evaluate_Weka_classifier(REPTree_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  REPTree_Test<-table( predict(REPTree_Classifier,newdata=test_set),test_set$target )
  if(!exists("REPTree_Test")){REPTree_Test<-summary(ZeroR_Classifier)}
  if(!exists("REPTree_Train")){REPTree_Train<-summary(ZeroR_Classifier)}
  if(!exists("REPTree_CV")){REPTree_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============DecisionStump===================
  DecisionStump_Classifier<-DecisionStump(train_file_clean$target~ ., data = train_file_clean)
  DecisionStump_Train<-summary(DecisionStump_Classifier)
  #Cross Validation
  DecisionStump_CV <- evaluate_Weka_classifier(DecisionStump_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  DecisionStump_Test<-table( predict(DecisionStump_Classifier,newdata=test_set),test_set$target )
  if(!exists("DecisionStump_Test")){DecisionStump_Test<-summary(ZeroR_Classifier)}
  if(!exists("DecisionStump_Train")){DecisionStump_Train<-summary(ZeroR_Classifier)}
  if(!exists("DecisionStump_CV")){DecisionStump_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============PART===================
  PART_Classifier<-PART(train_file_clean$target~ ., data = train_file_clean)
  PART_Train<-summary(PART_Classifier)
  #Cross Validation
  PART_CV <- evaluate_Weka_classifier(PART_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  PART_Test<-table( predict(PART_Classifier,newdata=test_set),test_set$target )
  if(!exists("PART_Test")){PART_Test<-summary(ZeroR_Classifier)}
  if(!exists("PART_Train")){PART_Train<-summary(ZeroR_Classifier)}
  if(!exists("PART_CV")){PART_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  
  
  
  #=============Table Models to choose Bagging model ==============
  Models<-c("ZeroR","OneR","BayesNet","DecisionStump","IBK","J48","LMT","Logistic","MultilayerPerceptron","NaiveBayes","PART","RandomForest","RandomTree","REPTree","SMO")
  FALSE_Correct_Clasified<-c(ZeroR_Train$confusionMatrix[2,2],OneR_Train$confusionMatrix[2,2],BayesNet_Train$confusionMatrix[2,2],DecisionStump_Train$confusionMatrix[2,2],IBK_Train$confusionMatrix[2,2],J48_Train$confusionMatrix[2,2],LMT_Train$confusionMatrix[2,2],Logistic_Train$confusionMatrix[2,2],MultilayerPerceptron_Train$confusionMatrix[2,2],NaiveBayes_Train$confusionMatrix[2,2],PART_Train$confusionMatrix[2,2],RandomForest_Train$confusionMatrix[2,2],RandomTree_Train$confusionMatrix[2,2],REPTree_Train$confusionMatrix[2,2],SMO_Train$confusionMatrix[2,2])
  FALSE_Correct_Clasified_CV<-c(ZeroR_CV$confusionMatrix[2,2],OneR_CV$confusionMatrix[2,2],BayesNet_CV$confusionMatrix[2,2],DecisionStump_CV$confusionMatrix[2,2],IBk_CV$confusionMatrix[2,2],J48_CV$confusionMatrix[2,2] ,LMT_CV$confusionMatrix[2,2],Logistic_CV$confusionMatrix[2,2],MultilayerPerceptron_CV$confusionMatrix[2,2],NaiveBayes_CV$confusionMatrix[2,2],PART_CV$confusionMatrix[2,2],RandomForest_CV$confusionMatrix[2,2],RandomTree_CV$confusionMatrix[2,2],REPTree_CV$confusionMatrix[2,2],SMO_CV$confusionMatrix[2,2])
  FALSE_Correct_Clasified_Test<-c(ZeroR_Test[2,2],OneR_Test[2,2],BayesNet_Test[2,2],DecisionStump_Test[2,2],IBk_Test[2,2],J48_Test[2,2] ,LMT_Test[2,2],Logistic_Test[2,2],MultilayerPerceptron_Test[2,2],NaiveBayes_Test[2,2],PART_Test[2,2],RandomForest_Test[2,2],RandomTree_Test[2,2],REPTree_Test[2,2],SMO_Test[2,2])
  TRUE_Correct_Clasified<-c(ZeroR_Train$confusionMatrix[1,1],OneR_Train$confusionMatrix[1,1],BayesNet_Train$confusionMatrix[1,1],DecisionStump_Train$confusionMatrix[1,1],IBK_Train$confusionMatrix[1,1],J48_Train$confusionMatrix[1,1],LMT_Train$confusionMatrix[1,1],Logistic_Train$confusionMatrix[1,1],MultilayerPerceptron_Train$confusionMatrix[1,1],NaiveBayes_Train$confusionMatrix[1,1],PART_Train$confusionMatrix[1,1],RandomForest_Train$confusionMatrix[1,1],RandomTree_Train$confusionMatrix[1,1],REPTree_Train$confusionMatrix[1,1],SMO_Train$confusionMatrix[1,1])
  TRUE_Correct_Clasified_CV<-c(ZeroR_CV$confusionMatrix[1,1],OneR_CV$confusionMatrix[1,1] ,BayesNet_CV$confusionMatrix[1,1],DecisionStump_CV$confusionMatrix[1,1],IBk_CV$confusionMatrix[1,1],J48_CV$confusionMatrix[1,1] ,LMT_CV$confusionMatrix[1,1],Logistic_CV$confusionMatrix[1,1],MultilayerPerceptron_CV$confusionMatrix[1,1],NaiveBayes_CV$confusionMatrix[1,1],PART_CV$confusionMatrix[1,1],RandomForest_CV$confusionMatrix[1,1],RandomTree_CV$confusionMatrix[1,1],REPTree_CV$confusionMatrix[1,1],SMO_CV$confusionMatrix[1,1])
  TRUE_Correct_Clasified_Test<-c(ZeroR_Test[1,1],OneR_Test[1,1] ,BayesNet_Test[1,1],DecisionStump_Test[1,1],IBk_Test[1,1],J48_Test[1,1] ,LMT_Test[1,1],Logistic_Test[1,1],MultilayerPerceptron_Test[1,1],NaiveBayes_Test[1,1],PART_Test[1,1],RandomForest_Test[1,1],RandomTree_Test[1,1],REPTree_Test[1,1],SMO_Test[1,1])
  #Build table models 
  Table_Models<-data.frame(Models,FALSE_Correct_Clasified,FALSE_Correct_Clasified_CV,TRUE_Correct_Clasified,TRUE_Correct_Clasified_CV,FALSE_Correct_Clasified_Test,TRUE_Correct_Clasified_Test)
  #True Possitive and Negatives
  TN<-summary(train_file_clean$target)[2]#True Negative
  TP<-summary(train_file_clean$target)[1]#True Positive
  TN_Test<-summary(test_set$target)[2]#True Negative
  TP_Test<-summary(test_set$target)[1]#True Positive
  #Accuracy
  Table_Models$Accuracy<-(FALSE_Correct_Clasified+TRUE_Correct_Clasified)/(TN+TP)
  Table_Models$Accuracy_Cross_Val<-(FALSE_Correct_Clasified_CV+TRUE_Correct_Clasified_CV)/(TN+TP)
  Table_Models$Accuracy_Test<-(FALSE_Correct_Clasified_Test+TRUE_Correct_Clasified_Test)/(TN_Test+TP_Test)
  #Build Sensitivity
  Table_Models$Sensitivity<-(TRUE_Correct_Clasified/TP)*100
  Table_Models$Sensitivity_CV<-(TRUE_Correct_Clasified_CV/TP)*100
  Table_Models$Sensitivity_Test<-(TRUE_Correct_Clasified_Test/TP_Test)*100
  #Build Specificity
  Table_Models$Specificity<-(FALSE_Correct_Clasified/TN)*100
  Table_Models$Specificity_CV<-(FALSE_Correct_Clasified_CV/TN)*100
  Table_Models$Specificity_Test<-(FALSE_Correct_Clasified_Test/TN_Test)*100
  #Build Overfitting
  Table_Models$Overfitting_Acc_vs_CV<-(Table_Models$Accuracy-Table_Models$Accuracy_Cross_Val)*100
  Table_Models$Overfitting_Acc_vs_Test<-(Table_Models$Accuracy-Table_Models$Accuracy_Test)*100
  #Add identifier for Simplest methods
  Table_Models$Ensamble<-ifelse(Table_Models$Models=="ZeroR"|Table_Models$Models=="OneR",-1,0)
  #Sort by best Methods
  #Sort by Accuracy
  Table_Models <- Table_Models[order(Table_Models$Accuracy),] 
  #Reassign Rows numbers to order
  rownames(Table_Models) <- NULL
  #Assign the column number to a new column
  Table_Models$Order_Accuracy<-rownames(Table_Models)
  #Sort by Accuracy CV
  Table_Models <- Table_Models[order(Table_Models$Accuracy_Cross_Val),] 
  rownames(Table_Models) <- NULL
  Table_Models$Order_Cross_Accuracy<-rownames(Table_Models)
  #Sort by Accuracy Test
  Table_Models <- Table_Models[order(Table_Models$Accuracy_Test),] 
  rownames(Table_Models) <- NULL
  Table_Models$Order_Test<-rownames(Table_Models)
  #Convert to numberic values to sum and order by the total
  Table_Models$Order_Accuracy<-as.numeric(Table_Models$Order_Accuracy)
  Table_Models$Order_Cross_Accuracy<-as.numeric(Table_Models$Order_Cross_Accuracy)
  Table_Models$Order_Test<-as.numeric(Table_Models$Order_Test)
  #Sort by Top
  Table_Models$Top<-Table_Models$Order_Cross_Accuracy+Table_Models$Order_Accuracy+Table_Models$Order_Test
  Table_Models$Top<-as.numeric(Table_Models$Top)
  Table_Models <- Table_Models[order(-Table_Models$Top),] 
  rownames(Table_Models) <- NULL
  Table_Models$Top<-rownames(Table_Models)
  Table_Models$Top<-as.numeric(Table_Models$Top)
  Table_Models<-subset(Table_Models, select = c(-20,-21,-22))
  return(Table_Models)
}
save.image()


Auto_ML_Bag_Bos_Ens<-function(Data,Size,Model1,Model2,Previous_Table){
  
  #============ZeroR function for errors
  ZeroR<-make_Weka_classifier("weka/classifiers/rules/ZeroR")
  
  #============Resampling data ================
  
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  
  
  #===============Bagging===================
  train_file_clean<-Data
  #train_file_clean<-resample(target~ .,data=train_file,control=Weka_control(Z=Size))
  
  #"weka.classifiers.trees.RandomForest"
  Best_Model_1<-Model1
  Best_Model_2<-Model2
  
  Bagging_Classifier<-Bagging(train_file_clean$target~ ., data = train_file_clean, control = Weka_control(W=Best_Model_1), na.action=NULL)
  Bagging_Train<-summary(Bagging_Classifier)
  #Cross Validation
  Bagging_CV <- evaluate_Weka_classifier(Bagging_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  if(!exists("Bagging_Train")){Bagging_Train<-summary(ZeroR_Classifier)}
  if(!exists("Bagging_CV")){Bagging_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  
  #===============LogitBoost===================
  
  #tryCatch(expr = {LogitBoost_Classifier<-LogitBoost(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(W=Best_Model_1),na.action=NULL)
  #},finally = {LogitBoost_Classifier<-ZeroR(train_file_clean$target~ ., data = train_file_clean)})
  #  LogitBoost_Train<-summary(LogitBoost_Classifier)
  #Cross Validation
  #LogitBoost_CV <- evaluate_Weka_classifier(LogitBoost_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #if(!exists("LogitBoost_Train")){LogitBoost_Train<-summary(ZeroR_Classifier)}
  #if(!exists("LogitBoost_CV")){LogitBoost_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============AdaBoostM1===================
  AdaBoostM1_Classifier<-AdaBoostM1(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(W=Best_Model_1), na.action=NULL)
  AdaBoostM1_Train<-summary(AdaBoostM1_Classifier)
  #Cross Validation
  AdaBoostM1_CV <- evaluate_Weka_classifier(AdaBoostM1_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  if(!exists("AdaBoostM1_Train")){AdaBoostM1_Train<-summary(ZeroR_Classifier)}
  if(!exists("AdaBoostM1_CV")){AdaBoostM1_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============Stacking===================
  Stacking_Classifier<-Stacking(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(
    M=Best_Model_1,
    B=Best_Model_2  ), na.action=NULL)
  Stacking_Train<-summary(Stacking_Classifier)
  #Cross Validation
  Stacking_CV <- evaluate_Weka_classifier(Stacking_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  if(!exists("Stacking_Train")){Stacking_Train<-summary(ZeroR_Classifier)}
  if(!exists("Stacking_CV")){Stacking_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #==============Joint Models=================
  Models<-c("Bagging","LogitBoost","AdaBoostM1","Stacking")
  FALSE_Correct_Clasified<-c(Bagging_Train$confusionMatrix[2,2],LogitBoost_Train$confusionMatrix[2,2],AdaBoostM1_Train$confusionMatrix[2,2],Stacking_Train$confusionMatrix[2,2])
  FALSE_Correct_Clasified_CV<-c(Bagging_CV$confusionMatrix[2,2],LogitBoost_CV$confusionMatrix[2,2],AdaBoostM1_CV$confusionMatrix[2,2],Stacking_CV$confusionMatrix[2,2])
  TRUE_Correct_Clasified<-c(Bagging_Train$confusionMatrix[1,1],LogitBoost_Train$confusionMatrix[1,1],AdaBoostM1_Train$confusionMatrix[1,1],Stacking_Train$confusionMatrix[1,1])
  TRUE_Correct_Clasified_CV<-c(Bagging_CV$confusionMatrix[1,1],LogitBoost_CV$confusionMatrix[1,1],AdaBoostM1_CV$confusionMatrix[1,1],Stacking_CV$confusionMatrix[1,1])
  Table_Models<-data.frame(Models,FALSE_Correct_Clasified,TRUE_Correct_Clasified,FALSE_Correct_Clasified_CV,TRUE_Correct_Clasified_CV)
  TN<-summary(train_file_clean$target)[2]#True Negative
  TP<-summary(train_file_clean$target)[1]#True Positive
  Table_Models$Accuracy<-((FALSE_Correct_Clasified+TRUE_Correct_Clasified)/(TN+TP))*100
  Table_Models$Cross_Val_Accuracy<-((FALSE_Correct_Clasified_CV+TRUE_Correct_Clasified_CV)/(TN+TP))*100
  Table_Models$Sensitivity<-(TRUE_Correct_Clasified/TP)*100
  Table_Models$Sensitivity_CV<-(TRUE_Correct_Clasified_CV/TP)*100
  Table_Models$Specificity<-(FALSE_Correct_Clasified/TN)*100
  Table_Models$Specificity_CV<-(FALSE_Correct_Clasified_CV/TN)*100
  Table_Models$Overfitting<-(Table_Models$Accuracy-Table_Models$Cross_Val_Accuracy)*100
  Table_Models$Ensamble<-ifelse(Table_Models$Models=="ZeroR"|Table_Models$Models=="OneR",-1,ifelse(Table_Models$Models=="Bagging"|Table_Models$Models=="LogitBoost"|Table_Models$Models=="AdaBoostM1"|Table_Models$Models=="Stacking",1,0))
  Table_Models<-rbind(Table_Models,Previous_Table)
  Table_Models<-Table_Models[order(Table_Models$Ensamble,Table_Models$Overfitting,Table_Models$FALSE_Correct_Clasified_CV),]
  rownames(Table_Models) <- NULL
  return(Table_Models)
}

save.image()
