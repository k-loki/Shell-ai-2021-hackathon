# Shell-ai-2021-hackathon
This is 35th place solution to shell ai hackathon hosted on HackerEarth platform.

[Link to the challenge here](https://www.hackerearth.com/challenges/competitive/shell-ai-hackathon-2021/instructions/)


The story:

- Shell.ai is a company focused on producing renewable and sustainable Energy. It got some solar fie
- But there are problems with solar Energy the sky can be cloudy or it may be raining or the weather is stormy, In these type of conditions the output of energy     
  produced may drastically go down.
- But our clients (shellai consumers) don't want drop in their energy consumption.
- So to manage the fluctuations in the energy output, we need to figure out the cloud cover in the future.
- This is where we as Data scientists come in, In this challenge we help shell.ai predict the cloud cover in the next 30 min, 60 min, 90 min and 120 min.
- These predictions may help the company cushion the impact of energy fluctuations.

My thoughts and approaches:

 - The training data provided information from different weather sensors like humidity, temperature, pressure, wind speed etc... and pictures of sky along with total        cloud cover percentage.
 - The target is total cloud cover in % of the sky.
 - I modeled the only tabular data without images cause images takes lot more compute power.
 - I cleaned the messy features.
 - Made interpolations to fill missing data without data leakage.
 - Used chained modelling
 - I finetuned random forests and LinearSVR to get better predictions. (Grid searching took a lot of time)
 - The loss metric is mean absolute Error (mae)
 - I tried ensembles of ensemble models like voting regressors but they didn't work out.
 - My best submissions are from fine-tuned Randomforest models.


Thank you :D

<!-- ![shell ai hackathon certificate_compressed](https://user-images.githubusercontent.com/84267959/170216423-cc9cd6e0-e223-4dc7-9daf-b24edd9a2201.jpg) -->
