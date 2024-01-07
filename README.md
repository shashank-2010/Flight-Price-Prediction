



![images](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/e84c0ec6-3706-4f84-ae59-56e5df14a131)

The Problem: Soaring into the future, the allure of travel can be met with the sting of sticker shock. Predicting flight prices remains an intricate dance, influenced by a complex interplay of factors. Airlines navigate a volatile landscape, balancing revenue with demand, fuel costs, and competition. For travelers, deciphering this riddle can mean the difference between a budget-friendly escape or a grounded dream. This project plunges into the world of machine learning, aiming to build a model that unravels the mysteries of flight pricing and empowers travelers with intelligent predictions.

Dataset - 
![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/113fa2e1-1734-4799-8d33-8cc2ab512164)

Data Cleaning
-------------

Data, the fuel of machine learning, can be messy. Before models feast, it needs cleaning. Inconsistent formats, missing values, and lurking errors taint the truth. We scrub away typos, standardize units, and fill in gaps with care. Outliers, those bizarre bumps, get gently nudged or banished. The cleaned data, now pure and pristine, feeds models with clarity, boosting their learning and unlocking accurate insights. Data, once muddled, now shines, powering predictions that soar. It includes removing null data, duplicated data, filling the data.

![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/7185458a-a8b4-442a-96ab-f9ff60717f11)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Data Analysis and Feature Analysis
-----------------------------------
Features, the raw ingredients of machine learning, hold hidden secrets. Untransformed, they're cryptic whispers to the models. Feature transformation casts a spell, reshaping and scaling, coaxing out meaning. Numbers dance in new scales, categories shed their robes, and words weave magic spells of vectors. Dimensions bend, revealing hidden patterns, like constellations in the data's dark sky. Transformed features, now clear and potent, ignite the models, feeding them knowledge and fueling accurate predictions.
![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/23d3720f-12c8-49e1-a736-cdb7f61aff57)

Graph to showcase the frequency of flight departure during a day
![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/517e8cf5-60b0-4eaf-93e6-012ca3ddb866)

Interactive Graphs - using plotly
![newplot (1)](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/33849a7d-2200-4abd-b4ad-b3d5cde1921a)

Feature Encoding
---------------
1. One Hot Encoding - Each unique category, no matter its hue, gets its own binary light – a 1 ablaze, others cloaked in 0s. 
![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/076eb394-1dce-4d06-9638-4dcf1f002155)

2. Target Guided Encoding
![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/1e61d1de-3e7a-4085-94ad-bae6cf80d7b0)

Handling Outliers - Using IQR
-----------------------------
![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/16dca3ed-e114-412b-8db8-d2230726645f)

Feature Selection
-----------------
In the realm of machine learning, features are the raw materials – the data points your model uses to build knowledge. But a plethora of features can be a curse, not a blessing. Irrelevant or redundant ones muddle the waters, masking the true relationships between features and outcomes. This is where feature selection steps in, acting as a meticulous sorter, discarding the dross and revealing the valuable insights hidden within.

Building Model
--------------
Random Forest Regressor
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
It is imported from sklearn.ensemble module.

Evaluation of Model
-------------------
Model evaluation is the process that uses some metrics which help us to analyze the performance of the model. As we all know that model development is a multi-step process and a check should be kept on how well the model generalizes future predictions. Therefore evaluating a model plays a vital role so that we can judge the performance of our model. The evaluation also helps to analyze a model’s key weaknesses. There are many metrics like Accuracy, Precision, Recall, F1 score, Area under Curve, Confusion Matrix, and Mean Square Error. Cross Validation is one technique that is followed during the training phase and it is a model evaluation technique as well.
![image](https://github.com/shashank-2010/Flight-Price-Prediction/assets/153171192/62538b33-5ead-466e-8118-100220b9e6bc)

------------------


