#### Exercise 1 
Read the data from the file `smoking.txt`, and divide the dataset into two groups consisting of smokers and non-smokers. 
Write a script which computes the average lung function, measured in FEV1, among the smokers and among the non-smokers. 
Report your computed average FEV1 scores. Are you surprised? 

##### Results. 
The measure of average lung function for smokers and non-smokers was **3.2768615384615387** and **2.5661426146010164** FEV1, respectively. 
This is quite surprising as smokers are shown to have a higher FEV1 average and the higher the FEV1, the higher the lung functioning. 
Smokers should have impaired respiration which should lead to a lower average. 
It is always important to be skeptical when relying on a mean as it is heavily affected by outliers that skew the distribution, 
however, the medians are 3.169 and 2.465, respectively, which also indicate that smokers have a higher overall lung functioning. 
On the other hand, other variables like age, might play a role in the observed functioning as smokers had a median age of 13 and 
non-smokers as 9- and 13-year-olds have more lung capacity.

#### Exercise 2
Make a box plot of the FEV1 in the two groups. What do you see? Are you surprised? 

##### Results. 
The box-plot indicates that the smoker and non-smoker lung function (FEV1) medians are approximately 2.5 and 3.1, respectively. 
The more interesting aspects of this plot indicate that there is a much larger range (approximately 1-4.8 FEV1) for non-smokers 
and notable outliers beyond 4.8 FEV1 while smokers have a condensed range (approx. 1.8-5 FEV1). This indicates that there is 
more variability within non-smokers which is proven to be the case as non-smokers and smokers have variances of 0.554 and 0.722, 
respectively. 

![GitHub Logo](/images/logo.png)
