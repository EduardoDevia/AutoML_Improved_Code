#===============================================================================
#====== Create a function to run Several ML Models for Classification ==========
#===============================================================================


AutoML<-function(DataFrame,Split_Value,Size,SMOTE){

  #==================Load Libraries to use Weka functions=======================
  #Library to run ML Models
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
  #Java Library
  library(rJava)
  #==================Resampling data ===========================================
  #Load weka filter
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  #Resample the data with the Size given
  DataFrame<-resample(target~ .,data=DataFrame,control=Weka_control(Z=Size))
  
  #==================Normalization ============================================
  #Select the numberic columns to be normalized
  columns_to_change<-colnames(select_if(DataFrame, is.numeric))
  #Dataframe to record mean and standard deveation
  Normalization_Values_mean_sd<-data.frame(Attribute=double(),Mean=double(),SDev=double())
  #Normalize Dataframe
  for(i in 1:length(columns_to_change)){
    column_number<-which(colnames(DataFrame)==columns_to_change[i])  
    #n1 - standardization ((x-mean)/sd)
    Normalization_Values_mean_sd[nrow(Normalization_Values_mean_sd) + 1,] <-list(
      Attribute=columns_to_change[i],Mean=mean(DataFrame[,column_number]),SDev=sd(DataFrame[,column_number]))
    
    DataFrame[,column_number]<-data.Normalization (
      DataFrame[,column_number] ,type="n1",normalization="column")
    
    }

  #==================Split data ==============================================
  #Create a new dataframe
  data=DataFrame
  #Create a split data with a split feature
  split = sample.split(data$target, SplitRatio = Split_Value)
  #Create training set
  training_set = subset(data, split == TRUE)
  #Create testing set
  test_set = subset(data, split == FALSE)
  
  #==================Split data ==============================================
    
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
  #DesicionStump 
  #J48
  #LMT
  #randomForest        
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
  Table_Models$Accuracy<-((FALSE_Correct_Clasified+TRUE_Correct_Clasified)/(TN+TP))*100
  Table_Models$Accuracy_Cross_Val<-((FALSE_Correct_Clasified_CV+TRUE_Correct_Clasified_CV)/(TN+TP))*100
  Table_Models$Accuracy_Test<-((FALSE_Correct_Clasified_Test+TRUE_Correct_Clasified_Test)/(TN_Test+TP_Test))*100
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
  .jcache(MultilayerPerceptron_Classifier$classifier)
  data_list<-list(Table_Models,Normalization_Values_mean_sd,
                  MultilayerPerceptron_Classifier,
                  J48_Classifier,
                  IBk_Classifier,
                  BayesNet_Classifier,
                  NaiveBayes_Classifier,
                  Logistic_Classifier,
                  SMO_Classifier,
                  LMT_Classifier,
                  RandomForest_Classifier,
                  RandomTree_Classifier,
                  REPTree_Classifier,
                  DecisionStump_Classifier,
                  PART_Classifier)
  return(data_list)
  
}
save.image()

#===============================================================================
#===================Bagging-Boosting-Ensamble Methods===========================
#===============================================================================

Auto_ML_Bag_Bos_Ens<-function(DataFrame,Split_Value,Size,Model1,Model2,Previous_Table,SMOTE){
  
  
  #==================Resampling data ===========================================
  #Load weka filter
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  #Resample the data with the Size given
  DataFrame<-resample(target~ .,data=DataFrame,control=Weka_control(Z=Size))
  
  #==================Normalization ============================================
  #Select the numberic columns to be normalized
  columns_to_change<-colnames(select_if(DataFrame, is.numeric))
  #Normalize Dataframe
  for(i in 1:length(columns_to_change)){column_number<-which(colnames(DataFrame)==columns_to_change[i])  
  DataFrame[,column_number]<-data.Normalization (DataFrame[,column_number] ,type="n1",normalization="column")}
  
  #==================Split data ==============================================
  #Create a new dataframe
  data=DataFrame
  #Create a split data with a split feature
  split = sample.split(data$target, SplitRatio = Split_Value)
  #Create training set
  training_set = subset(data, split == TRUE)
  #Create testing set
  test_set = subset(data, split == FALSE)
  
  #==================Split data ==============================================
  
  if (SMOTE=='Y') {
    #str(training_set$target)
    prop.table(table(training_set$target))
    training_set<-SMOTE(target ~ ., training_set, perc.over = 100, perc.under=200)
    #prop.table(table(training_set$target))
    
  }
  #=====================create a new file to use in the models==============
  train_file_clean<-training_set
  #========Load models to improve accuracy and reduce overfitting===========
  Best_Model_1<-Model1
  Best_Model_2<-Model2
  
  #=========================Bagging=========================================
  #Build the classifier
  Bagging_Classifier<-Bagging(train_file_clean$target~ ., data = train_file_clean, control = Weka_control(W=Best_Model_1), na.action=NULL)
  #summary to Evaluate the classifier
  Bagging_Train<-summary(Bagging_Classifier)
  #Cross Validation
  Bagging_CV <- evaluate_Weka_classifier(Bagging_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  Bagging_Test<-table( predict(Bagging_Classifier,newdata=test_set),test_set$target )
  if(!exists("Bagging_Train")){Bagging_Train<-summary(ZeroR_Classifier)}
  if(!exists("Bagging_CV")){Bagging_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  
  #===============AdaBoostM1===================
  AdaBoostM1_Classifier<-AdaBoostM1(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(W=Best_Model_1), na.action=NULL)
  AdaBoostM1_Train<-summary(AdaBoostM1_Classifier)
  #Cross Validation
  AdaBoostM1_CV <- evaluate_Weka_classifier(AdaBoostM1_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  AdaBoostM1_Test<-table( predict(AdaBoostM1_Classifier,newdata=test_set),test_set$target )
  if(!exists("AdaBoostM1_Train")){AdaBoostM1_Train<-summary(ZeroR_Classifier)}
  if(!exists("AdaBoostM1_CV")){AdaBoostM1_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #===============Stacking===================
  Stacking_Classifier<-Stacking(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(
    M=Best_Model_1,
    B=Best_Model_2  ), na.action=NULL)
  Stacking_Train<-summary(Stacking_Classifier)
  #Cross Validation
  Stacking_CV <- evaluate_Weka_classifier(Stacking_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  Stacking_Test<-table( predict(Stacking_Classifier,newdata=test_set),test_set$target )
  if(!exists("Stacking_Train")){Stacking_Train<-summary(ZeroR_Classifier)}
  if(!exists("Stacking_CV")){Stacking_CV<-evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)}
  #==============Joint Models=================
  Models<-c("Bagging","AdaBoostM1","Stacking")
  
  
  
  
  FALSE_Correct_Clasified<-c(Bagging_Train$confusionMatrix[2,2],AdaBoostM1_Train$confusionMatrix[2,2],Stacking_Train$confusionMatrix[2,2])
  FALSE_Correct_Clasified_Test<-c(Bagging_Test[2,2],AdaBoostM1_Test[2,2],Stacking_Test[2,2])
  FALSE_Correct_Clasified_CV<-c(Bagging_CV$confusionMatrix[2,2],AdaBoostM1_CV$confusionMatrix[2,2],Stacking_CV$confusionMatrix[2,2])
  TRUE_Correct_Clasified_Test<-c(Bagging_Test[1,1],AdaBoostM1_Test[1,1],Stacking_Test[1,1])
  TRUE_Correct_Clasified<-c(Bagging_Train$confusionMatrix[1,1],AdaBoostM1_Train$confusionMatrix[1,1],Stacking_Train$confusionMatrix[1,1])
  TRUE_Correct_Clasified_CV<-c(Bagging_CV$confusionMatrix[1,1],AdaBoostM1_CV$confusionMatrix[1,1],Stacking_CV$confusionMatrix[1,1])
  Table_Models<-data.frame(Models,FALSE_Correct_Clasified,FALSE_Correct_Clasified_CV,TRUE_Correct_Clasified,TRUE_Correct_Clasified_CV,FALSE_Correct_Clasified_Test,TRUE_Correct_Clasified_Test)
  
  TN<-summary(train_file_clean$target)[2]#True Negative
  TP<-summary(train_file_clean$target)[1]#True Positive
  TN_Test<-summary(test_set$target)[2]#True Negative
  TP_Test<-summary(test_set$target)[1]#True Positive
  
  
  Table_Models$Accuracy<-((FALSE_Correct_Clasified+TRUE_Correct_Clasified)/(TN+TP))*100
  Table_Models$Accuracy_Cross_Val<-((FALSE_Correct_Clasified_CV+TRUE_Correct_Clasified_CV)/(TN+TP))*100
  Table_Models$Accuracy_Test<-(FALSE_Correct_Clasified_Test+TRUE_Correct_Clasified_Test)/(TN_Test+TP_Test)*100
  Table_Models$Sensitivity<-(TRUE_Correct_Clasified/TP)*100
  Table_Models$Sensitivity_CV<-(TRUE_Correct_Clasified_CV/TP)*100
  Table_Models$Sensitivity_Test<-(TRUE_Correct_Clasified_Test/TP_Test)*100
  Table_Models$Specificity<-(FALSE_Correct_Clasified/TN)*100
  Table_Models$Specificity_CV<-(FALSE_Correct_Clasified_CV/TN)*100
  Table_Models$Specificity_Test<-(FALSE_Correct_Clasified_Test/TN_Test)*100
  Table_Models$Overfitting_Acc_vs_CV<-(Table_Models$Accuracy-Table_Models$Accuracy_Cross_Val)
  Table_Models$Overfitting_Acc_vs_Test<-(Table_Models$Accuracy-Table_Models$Accuracy_Test)
  Table_Models$Ensamble<-ifelse(Table_Models$Models=="ZeroR"|Table_Models$Models=="OneR",-1,ifelse(Table_Models$Models=="Bagging"|Table_Models$Models=="AdaBoostM1"|Table_Models$Models=="Stacking",1,0))
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
  Table_Models<-subset(Table_Models, select = c(-20,-21,-22))
  Table_Models<-rbind(Table_Models,Previous_Table)
  .jcache(Bagging_Classifier$classifier)
  data_models<-list(Table_Models,Bagging_Classifier,AdaBoostM1_Classifier,Stacking_Classifier)
  return(data_models)
}

save.image()

#===============================================================================
#===========================Attribute Selection ================================
#===============================================================================

Attribute_Selection<-function(DataFrame,Model_to_Evaluate,Bottom_Attributes,Split_Value,Size,SMOTE){
  
  #======================Load Libraries to use select_if==========================
  library(plyr)
  library(dplyr)
  setwd("C:/Users/chedevia/Desktop/Test")
  #==================Load Libraries to use Weka functions========================
  #Library to run ML Models
  library(RWeka) 
  #==============Libraries needed to normalize the numeric values================
  library(cluster)
  library(MASS)
  library(clusterSim)
  #===================Library to use SMOTE to balance the classes================
  library(grid)
  library(DMwR) 
  #===========================Library to split values============================
  library(caTools)
  #===========================Library to graph===================================
  library(scales)
  library(ggplot2)
  #install.packages("formattable")
  library(formattable)
  
  #===========================Load Weka Classifiers===============================
  ZeroR<-make_Weka_classifier("weka/classifiers/rules/ZeroR")
  MultilayerPerceptron<-make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron")
  BayesNet<-make_Weka_classifier("weka/classifiers/bayes/BayesNet")
  NaiveBayes<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
  RandomForest<-make_Weka_classifier("weka/classifiers/trees/RandomForest")
  RandomTree<-make_Weka_classifier("weka/classifiers/trees/RandomTree")
  REPTree<-make_Weka_classifier("weka/classifiers/trees/REPTree")
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  
  #==================Resampling data ===========================================
  #Resample the data with the Size given
  DataFrame<-resample(target~ .,data=DataFrame,control=Weka_control(Z=Size))
  
  #==================Normalization ============================================
  #Select the numberic columns to be normalized
  columns_to_change<-colnames(select_if(DataFrame, is.numeric))
  #Normalize Dataframe
  for(i in 1:length(columns_to_change)){column_number<-which(colnames(DataFrame)==columns_to_change[i])  
  DataFrame[,column_number]<-data.Normalization (DataFrame[,column_number] ,type="n1",normalization="column")}
  
  #==================Split data ==============================================
  #Create a new dataframe
  data_to_split=DataFrame
  #Create a split data with a split feature
  split = sample.split(data_to_split$target, SplitRatio = Split_Value)
  #Create training set
  training_set = subset(data_to_split, split == TRUE)
  #Create testing set
  test_set = subset(data_to_split, split == FALSE)
  
  #==================Split data ==============================================
  
  if (SMOTE=='Y') {
    #str(training_set$target)
    prop.table(table(training_set$target))
    training_set<-SMOTE(target ~ ., training_set, perc.over = 100, perc.under=200)
    #prop.table(table(training_set$target))
    
  }
  
  
  #Load correlation
  Correlation_AE <- make_Weka_attribute_evaluator("weka/attributeSelection/CorrelationAttributeEval")
  #Run Correlation
  Correlation_AE_table<-as.data.frame(Correlation_AE(training_set$target~ . , data = training_set))
  #Create a column with the row name
  Correlation_AE_table$row<-rownames(Correlation_AE_table)
  #Rename Column
  names(Correlation_AE_table)[1]<-paste('Correlation_AE')
  
  #Run Gain Ratio
  Gain_Ratio_AE_table<-as.data.frame(GainRatioAttributeEval(training_set$target~ . , data = training_set))
  #Create a column with the row name
  Gain_Ratio_AE_table$row<-rownames(Gain_Ratio_AE_table)
  #Rename Column
  names(Gain_Ratio_AE_table)[1]<-paste('Gain_Ratio_AE_table')
  
  #Run Info Gain
  Info_Gain_AE_table<-as.data.frame(InfoGainAttributeEval(training_set$target~ . , data = training_set))
  #Create a column with the row name
  Info_Gain_AE_table$row<-rownames(Info_Gain_AE_table)
  #Rename column
  names(Info_Gain_AE_table)[1]<-paste('Info_Gain_AE_table')
  
  #Load Attribute selection
  Attribute_Selection<-make_Weka_filter("weka/filters/supervised/attribute/AttributeSelection")
  
  #Run Wrapper
  Wrapper_Sub<-as.data.frame(colnames(Attribute_Selection(target ~., data = training_set, control = Weka_control("E" = "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.lazy.IBk", "S" = "weka.attributeSelection.GreedyStepwise"))))
  #Rename Column
  names(Wrapper_Sub)[1]<-paste('row')
  #Create a column with the row name
  Wrapper_Sub$Wrapper_Sub<-as.numeric(rownames(Wrapper_Sub))
  #Sort by Rank
  Wrapper_Sub <- Wrapper_Sub[order(-Wrapper_Sub$Wrapper_Sub),] 
  #Rename rows
  row.names(Wrapper_Sub)<-NULL
  #Assign new rank 
  Wrapper_Sub$Wrapper_Sub<-as.numeric(rownames(Wrapper_Sub))
  
  #Run CFS Correlation-based Feature Selection
  CFS_SE<-as.data.frame(colnames(Attribute_Selection(target ~., data = training_set, control = Weka_control("E" = "weka.attributeSelection.CfsSubsetEval", "S" = "weka.attributeSelection.GreedyStepwise")) ))
  #Rename Column
  names(CFS_SE)[1]<-paste('row')
  #Create a column with the row name
  CFS_SE$CFS_SE<-as.numeric(rownames(CFS_SE))
  #Sort by Rank
  CFS_SE <- CFS_SE[order(-CFS_SE$CFS_SE),] 
  #Rename rows
  row.names(CFS_SE)<-NULL
  #Assign new rank 
  CFS_SE$CFS_SE<-as.numeric(rownames(CFS_SE))
  
  #Run OneR Attribute Selection
  OneR_AE <-as.data.frame(colnames(Attribute_Selection(target ~., data = training_set, control = Weka_control("E" = "weka.attributeSelection.OneRAttributeEval", "S" = "weka.attributeSelection.Ranker"))))
  #Rename Column
  names(OneR_AE)[1]<-paste('row')
  #Create a column with the row name
  OneR_AE$OneR_AE<-as.numeric(rownames(OneR_AE))
  #Sort by Rank
  OneR_AE <- OneR_AE[order(-OneR_AE$OneR_AE),] 
  #Rename rows
  row.names(OneR_AE)<-NULL
  #Assign new rank 
  OneR_AE$OneR_AE<-as.numeric(rownames(OneR_AE))
  
  #Run relief attribute evaluation
  Relief_AE <-as.data.frame(colnames(Attribute_Selection(target ~., data = training_set, control = Weka_control("E" = "weka.attributeSelection.ReliefFAttributeEval", "S" = "weka.attributeSelection.Ranker"))))
  #Rename Column
  names(Relief_AE)[1]<-paste('row')
  #Create a column with the row name
  Relief_AE$Relief_AE<-as.numeric(rownames(Relief_AE))
  #Sort by Rank
  Relief_AE <- Relief_AE[order(-Relief_AE$Relief_AE),] 
  #Rename rows
  row.names(Relief_AE)<-NULL
  #Assign new rank 
  Relief_AE$Relief_AE<-as.numeric(rownames(Relief_AE))
  
  #Run Symmetrical Uncert AttributeEval
  Symmetrical_Uncert_AE <-as.data.frame(colnames(Attribute_Selection(target ~., data = training_set, control = Weka_control("E" = "weka.attributeSelection.SymmetricalUncertAttributeEval", "S" = "weka.attributeSelection.Ranker"))))
  #Rename Column
  names(Symmetrical_Uncert_AE)[1]<-paste('row')
  #Create a column with the row name
  Symmetrical_Uncert_AE$Symmetrical_Uncert_AE<-as.numeric(rownames(Symmetrical_Uncert_AE))
  #Sort by Rank
  Symmetrical_Uncert_AE <- Symmetrical_Uncert_AE[order(-Symmetrical_Uncert_AE$Symmetrical_Uncert_AE),] 
  #Rename rows
  row.names(Symmetrical_Uncert_AE)<-NULL
  #Assign new rank 
  Symmetrical_Uncert_AE$Symmetrical_Uncert_AE<-as.numeric(rownames(Symmetrical_Uncert_AE))
  
  #join data frames
  Attribute_Table<-list(Symmetrical_Uncert_AE,Relief_AE,OneR_AE,CFS_SE,Wrapper_Sub,Info_Gain_AE_table,
                        Gain_Ratio_AE_table,Correlation_AE_table) %>%
    Reduce(function(dtf1,dtf2) left_join(dtf1,dtf2,by="row"), .)
  #Change na to zero
  Attribute_Table[is.na(Attribute_Table)] <- 0
  #Sum attributes
  Attribute_Table$Rank<-rowSums(Attribute_Table[,2:6])
  #Sort by Rank
  Attribute_Table<-Attribute_Table[order(-Attribute_Table$Rank),] 
  #Rename rows
  row.names(Attribute_Table)<-NULL
  #Update Rank
  Attribute_Table$Rank<-as.numeric(row.names(Attribute_Table))
  
  #Sum attributes
  Attribute_Table$Rank1<-rowSums(Attribute_Table[,7:9])
  #Sort by Rank
  Attribute_Table<-Attribute_Table[order(-Attribute_Table$Rank1),] 
  #Rename rows
  row.names(Attribute_Table)<-NULL
  #Update Rank
  Attribute_Table$Rank1<-as.numeric(row.names(Attribute_Table))
  
  #Sum Ranks
  Attribute_Table$Rank<-Attribute_Table$Rank+Attribute_Table$Rank1
  #Sort by Rank
  Attribute_Table<-Attribute_Table[order(Attribute_Table$Rank),] 
  #Rename rows
  row.names(Attribute_Table)<-NULL
  #Update Rank
  Attribute_Table$Rank<-as.numeric(row.names(Attribute_Table))
  #Remove column Rank1
  Attribute_Table$Rank1<-NULL
  #Remove the target row from the data frame
  Attribute_Table<-subset(Attribute_Table,Attribute_Table$row!='target')
  #Now the bottom 5 attributes will be remove one by one to find the optimal number of
  #attributes to eliminate
  #Create a list of attributes to remove
  
  
  Attributes_to_remove<-Attribute_Table[(nrow(Attribute_Table)-Bottom_Attributes):nrow(Attribute_Table),1]
  
  


  #Create a blank dataframe to record the results of the attribute selection
  Optimal_Attributes_table<-data.frame(Attribute_invalid=double(),RF_Accuracy=double(),
                                       RF_CV_Accuracy=double(),RF_Test_Accuracy=double(),
                                       "Acc-Test"=double(),"Acc-Acc_CV"=double())


  for (i in 1:6){
    
    training_set<-subset(training_set,select = c(-(which(colnames(training_set)==Attributes_to_remove[7-i]))))
    
    #==================================Model To evaluate===========================
    #Load the classifier
    #RandomForest<-make_Weka_classifier("weka/classifiers/trees/RandomForest")
    #Train the model
    Classifier<-Model_to_Evaluate(training_set$target~ ., data = training_set)
    #Summarize the results
    Summary_Train<-summary(Classifier)
    #Accuracy
    Accuracy<-Summary_Train$details[1]
    #Cross Validation with 10 folds
    Summary_CV <- evaluate_Weka_classifier(Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
    #Accuracy
    CV_Accuracy<-Summary_CV$details[1]
    #Test
    Summary_Test<-table( predict(Classifier,newdata=test_set),
                              test_set$target )
    #Accuracy
    Test_Accuracy<-(Summary_Test[1,1]+Summary_Test[2,2]
    )/(Summary_Test[1,1]+Summary_Test[2,2]+
         Summary_Test[2,1]+Summary_Test[2,1])*100
    #Record the results in the Dataframe
    Optimal_Attributes_table[nrow(Optimal_Attributes_table) + 1,] = list(
      Attribute_invalid=Attributes_to_remove[7-i],Accuracy=Accuracy,
      CV_Accuracy=CV_Accuracy,Test_Accuracy=Test_Accuracy,
      "Acc-Test"=Accuracy-Test_Accuracy,"Acc-Acc_CV"=Accuracy-CV_Accuracy) 
    
  }
  
  
  results<-list(Attributes_to_remove,Optimal_Attributes_table,Attribute_Table)
  return(results)
}
save.image()




#===============================================================================
#=============== Reduction of the size remove noise ============================
#===============================================================================

Reduce_noise<-function(DataFrame,Model_to_Evaluate,Split_Value,Size,SMOTE){
  
  #======================Load Libraries to use select_if==========================
  library(plyr)
  library(dplyr)
  setwd("C:/Users/chedevia/Desktop/Test")
  #==================Load Libraries to use Weka functions========================
  #Library to run ML Models
  library(RWeka) 
  #==============Libraries needed to normalize the numeric values================
  library(cluster)
  library(MASS)
  library(clusterSim)
  #===================Library to use SMOTE to balance the classes================
  library(grid)
  library(DMwR) 
  #===========================Library to split values============================
  library(caTools)
  #===========================Library to graph===================================
  library(scales)
  library(ggplot2)
  #install.packages("formattable")
  library(formattable)
  
  #Load Dataframe
  DataFrame<-Data
  #Load sample size
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  #Create a copy of the dataframe
  train_file_clean<-training_set
  
  #Create a dataframe to record the optimal size of the dataframe
  Optimal_Size_table<-data.frame(Data_Size=double(),AB_Accuracy=double(),
                                 AB_CV_Accuracy=double(),AB_Test_Accuracy=double(),
                                 "Acc-Test"=double(),"Acc-Acc_CV"=double())
  
  #Find the most optimal size of the file that reduces the overfitting.
  for (i in 4:10) {
    #Size start with 10%  
    Size<-i*10
    resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
    #Resample the data with the Size given
    train_file_clean<-resample(target~ .,data=train_file_clean,control=Weka_control(Z=Size))
    
    #====================================Model Build===========================
    Classifier<-Model_to_Evaluate(train_file_clean$target~ .,
                                      data = train_file_clean,control = Weka_control(W=Model1), 
                                      na.action=NULL)
    Train<-summary(Classifier)
    Accuracy<-Train$details[1]
    #Cross Validation
    CV <- evaluate_Weka_classifier(Classifier, numFolds = 10, 
                                              complexity = FALSE, seed = 1, class = TRUE)
    CV_Accuracy<-CV$details[1]
    Test<-table( predict(Classifier,newdata=test_set),test_set$target )
    
    Percentage<-c(Size)
    
    #Accuracy
    Test_Accuracy<-(Test[1,1]+Test[2,2])/(Test[1,1]+Test[2,2]+Test[2,1]+Test[2,1])*100
    #Record the results in the Dataframe
    Optimal_Size_table[nrow(Optimal_Size_table) + 1,] <-list(Data_Size=Percentage,Accuracy=Accuracy,
                                                             CV_Accuracy=  CV_Accuracy,Test_Accuracy=Test_Accuracy,
                                                             "Acc-Test"=Accuracy-Test_Accuracy,"Acc-Acc_CV"=Accuracy-CV_Accuracy)
  }
  #reducing the size of the data did not reduce the accuracy
  return(Test_Accuracy)
}

save.image()




#===============================================================================
#==== Create a function to run Several ML Models for numeric prediction ========
#===============================================================================






AutoML_Numeric<-function(DataFrame,Split_Value,Size,SMOTE){
  
  #==================Load Libraries to use Weka functions=======================
  #Library to run ML Models
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
  #Java Library
  library(rJava)
  #==================Resampling data ===========================================
  #Load weka filter
  resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")
  #Resample the data with the Size given
  DataFrame<-resample(target~ .,data=DataFrame,control=Weka_control(Z=Size))
  
  #==================Normalization ============================================
  #Select the numberic columns to be normalized
  columns_to_change<-colnames(select_if(DataFrame, is.numeric))
  #Dataframe to record mean and standard deveation
  Normalization_Values_mean_sd<-data.frame(Attribute=double(),Mean=double(),SDev=double())
  #Normalize Dataframe
  for(i in 1:length(columns_to_change)){
    column_number<-which(colnames(DataFrame)==columns_to_change[i])  
    #n1 - standardization ((x-mean)/sd)
    Normalization_Values_mean_sd[nrow(Normalization_Values_mean_sd) + 1,] <-list(
      Attribute=columns_to_change[i],Mean=mean(DataFrame[,column_number]),SDev=sd(DataFrame[,column_number]))
    
    DataFrame[,column_number]<-data.Normalization (
      DataFrame[,column_number] ,type="n1",normalization="column")
    
  }
  
  #==================Split data ==============================================
  #Create a new dataframe
  data=DataFrame
  #Create a split data with a split feature
  split = sample.split(data$target, SplitRatio = Split_Value)
  #Create training set
  training_set = subset(data, split == TRUE)
  #Create testing set
  test_set = subset(data, split == FALSE)
  
  #==================Split data ==============================================
  
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
  #================Prediction==============#
  #ZeroR
  #GaussianProcesses
  #LinearRegression
  #MultilayerPerceptron
  #SMOreg
  #IBk
  #KStar
  #LWL
  #Bagging
  #DecistionTable
  #M5Rules
  #DesicionStump 
  #M5P
  #RandomForest        
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
  Table_Models$Accuracy<-((FALSE_Correct_Clasified+TRUE_Correct_Clasified)/(TN+TP))*100
  Table_Models$Accuracy_Cross_Val<-((FALSE_Correct_Clasified_CV+TRUE_Correct_Clasified_CV)/(TN+TP))*100
  Table_Models$Accuracy_Test<-((FALSE_Correct_Clasified_Test+TRUE_Correct_Clasified_Test)/(TN_Test+TP_Test))*100
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
  .jcache(MultilayerPerceptron_Classifier$classifier)
  data_list<-list(Table_Models,Normalization_Values_mean_sd,
                  MultilayerPerceptron_Classifier,
                  J48_Classifier,
                  IBk_Classifier,
                  BayesNet_Classifier,
                  NaiveBayes_Classifier,
                  Logistic_Classifier,
                  SMO_Classifier,
                  LMT_Classifier,
                  RandomForest_Classifier,
                  RandomTree_Classifier,
                  REPTree_Classifier,
                  DecisionStump_Classifier,
                  PART_Classifier)
  return(data_list)
  
}
save.image()