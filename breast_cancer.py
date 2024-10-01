1. # Importing the necessary libraries and modules
  2. import os
  3. from sklearn import svm, linear_model, neighbors, neural_network, naive_bayes, ensemble
  4. from sklearn.ensemble import RandomForestClassifier
  5. from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
  6. from sklearn.model_selection import train_test_split
  7. from sklearn.preprocessing import StandardScaler
  8. from sklearn.neural_network import MLPClassifier
  9. import matplotlib.pyplot as plt
 10. import numpy as np
 11. import tkinter as tk
 12. #from tkinter import messagebox
 13. import pandas as pd
 14. import joblib
 15. import seaborn as sns
 16. from numpy import unique
 17. from PIL import Image, ImageTk
 18. from sklearn.mixture import GaussianMixture
 19. import random
 20.  
 21.  
 22. import matplotlib.backends.backend_tkagg as tkagg
 23. from matplotlib.figure import Figure
 24.  
 25. ############ DESCRIPTIVE STATISTICS ################
 26.  
 27. # Read data from local disk
 28. data = pd.read_csv('data1.csv')
 29.  
 30.  
 31. # Displaying descriptive statistics
 32. print("=====================================================")
 33. print("                                 DESCRIPTION        ")
 34. print("=====================================================")
 35. summary_stats = data.describe()
 36. print(summary_stats)
 37. summary_stats.to_csv("summary_stats.csv", index=True)
 38.  
 39.  
 40. # Counting missing values in each column
 41. missing_counts = data.isnull().sum()
 42. print("\n\n=====================================================")
 43. print("                                 MISSING VALUES        ")
 44. print("=====================================================")
 45. print(missing_counts)
 46.  
 47.  
 48. # Create a table for diagnosis frequencies
 49. diagnosis_table = pd.Series(data["diagnosis"]).value_counts()
 50.  
 51. # Create colors for the pie chart
 52. colors = ["#1f77b4", "#ff7f0e"]  
 53.  
 54. # Compute the proportion of each diagnosis category
 55. diagnosis_prop_table = diagnosis_table / len(data) * 100
 56.  
 57. # Create labels for the pie chart
 58. pielabels = [f"{diagnosis} - {proportion:.1f}%" for diagnosis, 
 59.              proportion in zip(diagnosis_prop_table.index, diagnosis_prop_table)]
 60.  
 61. # Create the pie chart
 62. plt.pie(diagnosis_prop_table, labels=pielabels, colors=colors, autopct="%1.1f%%",
 63.         startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'gainsboro'})
 64. plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
 65.  
 66. # Add a title and legend
 67. plt.title("Frequency of Cancer Diagnosis")
 68. plt.legend(diagnosis_prop_table.index, loc="center left", 
 69.            bbox_to_anchor=(1, 0.5), title="Diagnosis", title_fontsize="medium")
 70.  
 71. # Show the plot
 72. plt.show()
 73.  
 74.  
 75. # Create a frequency distribution table DataFrame
 76. frequency_table = pd.DataFrame({"Diagnosis": diagnosis_table.index,
 77.                                 "Frequency": diagnosis_table.values})
 78.  
 79. # Save the frequency distribution table to a .csv file
 80. frequency_table.to_csv("frequency_distribution.csv", index=False)
 81.  
 82.  
 83. # Calculate collinearity (correlation matrix)
 84. corMatMy = data.iloc[:, 1:31].corr()
 85.  
 86. # Set up the figure
 87. plt.figure(figsize=(12, 10))
 88.  
 89. # Create a custom diverging colormap
 90. cmap = sns.color_palette("coolwarm", as_cmap=True)
 91.  
 92. # Plot the correlation matrix using seaborn's heatmap
 93. sns.heatmap(corMatMy, annot=False, fmt=".2f", cmap=cmap, center=0,
 94.             square=True, linewidths=0.5, cbar_kws={"shrink": 0.7}, annot_kws={"size": 8})
 95.  
 96. # Customize the plot
 97. plt.title("Correlation Matrix", fontsize=16)
 98. plt.xticks(rotation=90)
 99. plt.yticks(rotation=0)
100. plt.tight_layout()
101.  
102. # Show the plot
103. plt.show()
104.  
105. # Split the DataFrame into 'diagnosis' and 'features' parts
106. diagnosis = data.iloc[:, 0]
107. features = data.iloc[:, 1:]
108.  
109. # Get the unique values in the 'diagnosis' column (assuming 'B' and 'M' are the possible values)
110. unique_diagnosis = diagnosis.unique()
111.  
112. # Plot histograms for all feature columns
113. for column in features.columns:
114.     plt.figure()
115.     for diag in unique_diagnosis:
116.         plt.hist(features.loc[diagnosis == diag, column], alpha=0.5, label=diag)
117.     plt.xlabel(column)
118.     plt.ylabel('Frequency')
119.     plt.legend()
120.     plt.show()
121.  
122.  
123. # Split data into features and target variable
124. X = data.drop('diagnosis', axis=1)
125. y = data['diagnosis']
126.     
127. ################ Scale the data###################   
128. scaler = StandardScaler()
129. X_scaled = scaler.fit_transform(X)
130.  
131. #########Split data into training and testing sets ############
132. X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
133.  
134. # Initializing the Models
135. svm_model = svm.SVC()
136. logistic_model = linear_model.LogisticRegression()
137. knn_model = neighbors.KNeighborsClassifier()
138. ann_model = neural_network.MLPClassifier()
139. naive_bayes_model = naive_bayes.GaussianNB()
140. random_forest_model = RandomForestClassifier()
141.  
142.  
143. ######################## Training the Models #################################
144.  
145. svm_model.fit(X_train, y_train)
146. logistic_model.fit(X_train, y_train)
147. knn_model.fit(X_train, y_train)
148. ann_model = MLPClassifier(max_iter=500, random_state=42)
149. ann_model.fit(X_train, y_train)
150. naive_bayes_model.fit(X_train, y_train)
151. random_forest_model.fit(X_train, y_train)
152.  
153.  
154. classifiers = [svm_model, logistic_model, knn_model, ann_model, naive_bayes_model, random_forest_model]
155. print("\n\n =======================================================================================\n")
156. print("        PERFORMANCE OF THE MODELS: CONFUSION MATRIX, ACCURACY, F1-SCORE AND PRECISION")
157. print("\n =======================================================================================\n")
158. i = 0
159. for classifier in classifiers:
160.     i +=1
161.     y_pred = classifier.predict(X_test)
162.     cm = confusion_matrix(y_test, y_pred)
163.     acc = round(accuracy_score(y_test, y_pred),4)
164.     f1 = round(f1_score(y_test, y_pred, pos_label='M'),4)
165.     precision= round(precision_score(y_test, y_pred, pos_label='M'),4)
166.      
167.     print(f"\t ({i}) Classifier: {classifier.__class__.__name__}")
168.     print("\t \t Confusion Matrix:")
169.     print("\t \t \t " + str(cm[0][:]))
170.     print("\t \t \t " + str(cm[1][:]))
171.     print("\t \t Accuracy:", acc)
172.     print("\t \t F1 Score:", f1)
173.     print("\t \t Precision:", precision)
174.     print("\n")
175.  
176. ##### Feature Importance Based on the Random Forest Model ####
177. rf_feature_importance = random_forest_model.feature_importances_
178.  
179. # create a table to display the feature importance
180. table = pd.DataFrame({'Feature': X.columns, 'Importance': rf_feature_importance})
181. table = table.sort_values('Importance', ascending=False)
182. print("\n\n =======================================================================================\n")
183. print("\t\t\t\t\t\t \t FEATURE IMPORTANCE")
184. print("\n =======================================================================================\n")
185.  
186. print(table)
187.  
188. print("\n\n\n =======================================================================================\n")
189. print("\t\t\t\t\t\t \t IGNORE THIS INCOMING WARNING")
190. print("\n =======================================================================================\n")
191.  
192. rf_feature_importance = random_forest_model.feature_importances_
193. ranks_and_features = zip(rf_feature_importance,X.columns)
194. ranks_and_features = sorted(ranks_and_features,reverse=True)
195. keys = [k[1] for k in ranks_and_features ] [::-1]
196. values = [k[0] for k in ranks_and_features ][::-1]
197. plt.figure(figsize=(12, 10))
198.  
199. plt.barh(keys, values)
200. plt.xlabel('Feature Importance')
201. plt.ylabel('Features')
202. #plt.title('Random Forest - Feature Importance')
203. plt.show()
204.     
205.  
206. ## Top four classifiers based on their performance
207. print("\n\n =======================================================================================\n")
208. print("\t\t\t\t\t\t \t THE FOUR BEST MODELS")
209. print("\n =======================================================================================\n")
210.  
211. top_models = [('SVM', svm_model),('Logistic Regression', logistic_model), ('Naive Bayes', naive_bayes_model),
212.                    ('Artificial Neural Netowrk',ann_model)]
213.  
214. for i in range(4):
215.     print("(" + str(i+1) + ")  " + str(top_models[i][0]) + " using the command " + str(top_models[i][1]))
216.  
217. ##### Initialize the Ensemble Model #####
218. ensemble_model = ensemble.VotingClassifier(estimators=top_models,
219.                                            voting='hard')
220.  
221. # Training the ensemble model with the top four models
222. ensemble_model.fit(X_train, y_train)
223.  
224. # Testing the ensemble model
225. ensemble_pred = ensemble_model.predict(X_test)
226.  
227. #Evaluation of the Ensemble Model
228. cm_ensemble = confusion_matrix(y_test, ensemble_pred)
229. acc_ensemble = round(accuracy_score(y_test, ensemble_pred),4)
230. f1_ensemble = round(f1_score(y_test, ensemble_pred, pos_label='M'),4)
231. precision_ensemble= round(precision_score(y_test, ensemble_pred, pos_label='M'),4)
232.  
233. print("\n\n =======================================================================================\n")
234. print("\t\t\t\t\t\t \t PERFORMANCE OF THE ENSEMBLE MODEL")
235. print("\n =======================================================================================\n")
236.  
237. print("\t \t Confusion Matrix:")
238. print("\t \t \t " + str(cm_ensemble[0][:]))
239. print("\t \t \t " + str(cm_ensemble[1][:]))
240. print("\t \t Accuracy:", acc_ensemble)
241. print("\t \t F1 Score:", f1_ensemble)
242. print("\t \t Precision:", precision_ensemble)
243. print("\n")
244.  
245.  
246. # List of classifiers
247. classifiers = [
248.     ('SVM', svm_model),
249.     ('Logistic Regression', logistic_model),
250.     ('KNN', knn_model),
251.     ('ANN', ann_model),
252.     ('Naive Bayes', naive_bayes_model),
253.     ('Random Forest', random_forest_model)
254. ]
255.  
256. # Lists to store performance metrics
257. accuracy_scores = []
258. f1_scores = []
259. precision_scores = []
260.  
261. # Create subplots for visualizing performance on an A4 size page
262. fig, axs = plt.subplots(len(classifiers), 2, figsize=(8.27, 11.69), constrained_layout=True)
263. plt.subplots_adjust(wspace=0.3, hspace=0.5)
264.  
265. for i, (classifier_name, classifier) in enumerate(classifiers):
266.     # Predictions
267.     y_pred = classifier.predict(X_test)
268.  
269.     # Performance metrics
270.     acc = round(accuracy_score(y_test, y_pred) * 100, 5)
271.     f1 = round(f1_score(y_test, y_pred, pos_label='M') * 100, 5)
272.     precision = round(precision_score(y_test, y_pred, pos_label='M') * 100, 5)
273.  
274.     # Store performance metrics
275.     accuracy_scores.append(acc)
276.     f1_scores.append(f1)
277.     precision_scores.append(precision)
278.  
279.     # Plot confusion matrix
280.     sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g', ax=axs[i, 0])
281.     axs[i, 0].set_title(f'{classifier_name} - Confusion Matrix')
282.     axs[i, 0].set_xlabel('Predicted')
283.     axs[i, 0].set_ylabel('Actual')
284.  
285.     # Plot bar chart for metrics
286.     metrics_names = ['Accuracy', 'F1 Score', 'Precision']
287.     metrics_values = [acc, f1, precision]
288.     axs[i, 1].bar(metrics_names, metrics_values, color=['lightblue', 'lightgreen', 'lightcoral'])
289.     axs[i, 1].set_title(f'{classifier_name} - Performance Metrics')
290.     axs[i, 1].set_ylabel('Score')
291.  
292. # Save the performance plots as an image
293. plt.savefig('classifier_performance.png', dpi=300, bbox_inches='tight')
294.  
295. # Show the plots
296. plt.show()
297.  
298. # Display a summary table of performance metrics
299. performance_summary = pd.DataFrame({
300.     'Classifier': [classifier_name for classifier_name, _ in classifiers],
301.     'Accuracy (%)': accuracy_scores,
302.     'F1 Score (%)': f1_scores,
303.     'Precision (%)': precision_scores
304. })
305.  
306. # Create a title for the table
307. title_text = 'Classifier Performance Summary'
308.  
309. # Display the summary table with larger size and adjusted font size
310. fig, ax = plt.subplots(figsize=(8.27, 2))
311. ax.axis('off')
312. table = ax.table(cellText=performance_summary.values, colLabels=performance_summary.columns, cellLoc='center', loc='center')
313. table.auto_set_font_size(False)
314. table.set_fontsize(12)
315. table.scale(1, 1.5)
316.  
317. # Add a title separately
318. plt.title(title_text, fontsize=14)
319.  
320. # Save the summary table as an image
321. plt.savefig('classifier_performance_summary.png', dpi=300, bbox_inches='tight')
322.  
323. # Show the summary table
324. plt.show()
325.  
326. # Print the performance summary in tabular format
327. print("\nPERFORMANCE SUMMARY")
328. print(performance_summary)
329.  
330. #### Create a DataFrame to store the new dataset from Ensemble Predictions
331. # Convert the string in the Ensemble Prediction to float and save as cluster_list
332. cluster_list = []
333. for i in range(len(ensemble_pred)):
334.     if ensemble_pred[i] == 'M':
335.         cluster_list.append(0)
336.     elif ensemble_pred[i] =='B':
337.         cluster_list.append(1)
338.  
339. ##### Create a new Dataset
340. new_dataset = pd.DataFrame(X_test, columns=X.columns)
341.  
342. # Add the predicted labels as a new column
343. new_dataset['predicted_diagnosis'] = cluster_list
344.  
345. #Remove existing .csv file so as to create space for the new .csv file
346. if os.path.exists("new_dataset3.csv"):
347.     os.remove("new_dataset3.csv") # one file at a time
348.  
349. # Save the new dataset to a CSV file
350. new_dataset.to_csv('new_dataset3.csv', index=False)
351.  
352. ##### Applying unsupervised clustering on the ensemble predictions
353. cluster_data = []
354. cluster_data = pd.read_csv("new_dataset3.csv")
355. cluster_data.head()
356.  
357. features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
358.        'smoothness_mean', 'compactness_mean', 'concavity_mean',
359.        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
360.        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
361.        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
362.        'fractal_dimension_se', 'radius_worst', 'texture_worst',
363.        'perimeter_worst', 'area_worst', 'smoothness_worst',
364.        'compactness_worst', 'concavity_worst', 'concave_points_worst',
365.        'symmetry_worst', 'fractal_dimension_worst']
366.  
367. data = cluster_data[features].copy()
368.  
369. data.head()
370.  
371. # define the model
372. n_components = 2
373. gaussian_model = GaussianMixture(n_components=n_components)
374.  
375. # train the model
376. gaussian_model.fit(data)
377.  
378. # assign each data point to a cluster
379. gaussian_result = gaussian_model.predict(data)
380.  
381. #### Evaluate the Gaussian Mixed Model on the Data #####
382. # Assess component weights and probabilities
383. weights = gaussian_model.weights_
384. probabilities = gaussian_model.predict_proba(data)
385.  
386. # Evaluate the model
387. bic = gaussian_model.bic(data)
388. aic = gaussian_model.aic(data)
389.  
390. # Plotting the data and estimated Gaussian components
391. x = np.linspace(-5, 10, 1000)
392. plt.figure(figsize=(10, 8))
393. plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
394. for i in range(n_components):
395.     mean = gaussian_model.means_[i][0]
396.     std = np.sqrt(gaussian_model.covariances_[i][0][0])
397.     plt.plot(x, weights[i] * np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2),
398.              label=f'Component {i+1}')
399. plt.legend()
400. plt.xlabel('Value')
401. plt.ylabel('Density')
402. #plt.title('GMM Fit to the Data')
403. plt.xlim(-3, 4.5)  # Set x-axis limits from 0 to 6
404. plt.ylim(0, 1.0) #Set y-axis limits from 0 to 6
405.  
406. plt.show()
407.  
408. print("\n\n =======================================================================================\n")
409. print("\t\t\t\t\t\t \t GAUSSIAN MODEL: WEIGHTS AND PROBABILITIES")
410. print("\n =======================================================================================\n")
411.  
412.  
413. # Print the results
414. print(f"\t \t Component 1 Weight = {round(weights[0],4)}\t \t  Component 2 Weight = {round(weights[1],4)}" )
415. print("\t \t Probabilities:")
416. for i in range(n_components):
417.     print(f"\t \t Component {i+1}: {probabilities[:3, i]}")
418. print("\n\t \t BIC:", bic)
419. print("\t \t AIC:", aic)
420.  
421. # get all of the unique clusters
422. gaussian_clusters = unique(gaussian_result)
423.  
424. predicted_diagnosis = cluster_data.predicted_diagnosis
425.  
426. # Create an empty list to store cluster dataframes
427. clusters = []
428.  
429. # Iterate over each cluster
430. print("\n\n =======================================================================================\n")
431. print("\t\t\t\t\t\t \t CLUSTERS")
432. print("\n =======================================================================================\n")
433. for i in gaussian_clusters:
434.     # Filter the cluster data based on predicted diagnosis and features
435.     cluster = cluster_data[predicted_diagnosis == i][["predicted_diagnosis"] + features]
436.     clusters.append(cluster)
437.  
438.     # Print cluster information
439.     count = cluster['predicted_diagnosis'].value_counts()
440.     percentage = cluster['predicted_diagnosis'].value_counts(normalize=True) * 100
441.     print(f"\t \t \t \t CLUSTER {i}")
442.     print("\t \t \t \t \t \t Count: ",count.to_string(header=False))
443.     print("\t \t \t \t \t Percentage:", percentage.to_string(header=False))
444.     print("\n")
445.     
446. # Calculate average features for each cluster
447. average_features = []
448. for i, cluster in enumerate(clusters):
449.     average = cluster[features].mean()
450.     average_features.append(average)
451.     average_df = pd.DataFrame(average_features, columns=features)
452.     #print (average_df)
453.  
454. # Create a scatter plot for cluster 0 and 1
455. plt.scatter(range(len(average_df.columns)), average_df.iloc[0], color='red', marker='o', label="Cluster 0")
456. plt.scatter(range(len(average_df.columns)), average_df.iloc[1], color='green', marker='o', label="Cluster 1")
457.  
458. # Add legend, labels, and title to the plot
459. plt.legend()
460. plt.xlabel("Features")
461. plt.ylabel("Average Feature Value")
462. #plt.title("Cluster Plot")
463.  
464. plt.show()
465.  
466. # AI implementation
467.          
468. # Save the Clustering Model
469. joblib.dump(gaussian_model, 'gaussian_clustering.joblib')
470.  
471.  
472. # Load the trained clustering model
473. model = joblib.load('gaussian_clustering.joblib')
474.  
475. # Function to perform cancer diagnosis prediction
476. def predict_diagnosis(data):
477.     # Perform clustering on the input data
478.     cluster_label = model.predict(data)
479.     
480.     # Map the cluster label to a diagnosis (Malignant or Benign)
481.     if cluster_label == 0:
482.         diagnosis = "Malignant"
483.     else:
484.         diagnosis = "Benign"
485.  
486.     # Return the predicted diagnosis
487.     return diagnosis
488.  
489. # Function to handle button click event
490. def predict_button_click():
491.     # Get input values from entry fields
492.     input_values = [float(entry.get()) for entry in entry_fields.values()]
493.  
494.     # Create a DataFrame from the input values
495.     input_data = pd.DataFrame([input_values], columns=entry_fields.keys())
496.  
497.     # Perform prediction using the trained clustering model
498.     result = predict_diagnosis(input_data)
499.     if result == "Malignant":
500.         cluster_label = 0
501.     else:
502.         cluster_label = 1
503.     
504.     print("\n",result, cluster_label,"\n")
505.  
506.     #Find distance using Carnberra distance
507.     data_list = average_df.values.tolist()
508.     input_data_list = input_data.values.tolist()
509.     percent_output = round(100-abs(100*np.corrcoef(input_data_list, data_list[cluster_label])[0, 1]),2)
510.  
511.     b = max(input_data_list[0])
512.     a = min(input_data_list[0])
513.     scale_factor = b-a
514.     input_data_list = [(y - a)/(scale_factor) for y in input_data_list[0]]
515.     b = max(data_list[cluster_label])
516.     a = min(data_list[cluster_label])
517.     scale_factor = b-a
518.     data_list = [(y-a)/(scale_factor) for y in data_list[cluster_label] ]      
519.         
520.     result_window = tk.Toplevel()
521.     result_window.title("Classfication")
522.     result_window.resizable(True, True)
523.     
524.     
525.     # Create a Figure for the Matplotlib plot
526.     fig = Figure(figsize=(6, 6))
527.     ax = fig.add_subplot(111)
528.     ax.scatter(range(len(data_list)), data_list, marker='*', color='red', label=result)
529.     ax.scatter(range(len(input_data_list)), input_data_list, marker='o', color='blue', label='New')
530.     ax.set_xlabel("Features")
531.     ax.set_ylabel(result + " Cluster and New Data")
532.     ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.13))
533.  
534.     # Create a Tkinter PhotoImage from the Matplotlib figure
535.     canvas = tkagg.FigureCanvasTkAgg(fig, master=result_window)
536.     canvas_widget = canvas.get_tk_widget()
537.     
538.     # Use pack to expand the canvas as the window is resized
539.     canvas_widget.pack(fill=tk.BOTH, expand=True)
540.     
541.     
542.     #Display the classified and percentages on the window
543.     result_label = tk.Label(result_window, text =  f"Cancer classification:  {percent_output}%  {result}", font=("Arial", 14))
544.     result_label.pack(pady=10)    
545.     
546.     # # Clear the canvas
547.     canvas.delete("plot")  # Delete any existing plot on the canvas
548.     
549.     # Plotting the graph on the canvas
550.     fig, ax = plt.subplots(figsize=(6, 6)) 
551.     ax.scatter(range(len(data_list)), data_list, marker='*', color='red',label=result)
552.     ax.scatter(range(len(input_data_list)), input_data_list, marker='o', color='blue', label = 'New')
553.     
554.     ax.set_xlabel("Features")
555.     ax.set_ylabel(result +" Cluster and New Data")
556.     #ax.set_title("Visual Representation: New Data vs "+ result + " Cluster")
557.     ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.13))   
558.     
559.     #Create a canvas in the plot window
560.     plot_canvas = tk.Canvas(result_window,width = 1000, height = 800)
561.     plot_canvas.pack()
562.     
563.     # Convert the Matplotlib figure to a Tkinter PhotoImage
564.     plot_image = fig
565.  
566.     # Display the plot image on the canvas
567.     plot_canvas.create_image(0.5, 0.5, anchor=tk.NW, image=plot_image, tags="plot")
568.  
569.  
570. # Create the application window
571. window = tk.Tk()
572. window.title("Breast Cancer Diagnosis System")
573. window.geometry("1200x1200")
574. window.resizable(True, True)
575.  
576. # Create a canvas to place the background image
577. canvas = tk.Canvas(window, width=900, height=900)
578. canvas.pack()
579.  
580. # Load the background image
581. bg_image = Image.open("img_cancer.jpg")
582. bg_photo = ImageTk.PhotoImage(bg_image)
583.  
584.  
585. # Place the background image on the canvas
586. canvas.create_image(0.5, 0.5, anchor=tk.NW, image=bg_photo)
587.  
588.  
589. # Create a frame for the input fields
590. input_frame = tk.Frame(window, bg='white')
591. input_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
592.  
593.  
594. # Create the input labels and entry fields
595. labels = [
596.             'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
597.             'smoothness_mean', 'compactness_mean', 'concavity_mean',
598.            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
599.            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
600.            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
601.            'fractal_dimension_se', 'radius_worst', 'texture_worst',
602.            'perimeter_worst', 'area_worst', 'smoothness_worst',
603.            'compactness_worst', 'concavity_worst', 'concave_points_worst',
604.            'symmetry_worst', 'fractal_dimension_worst'
605. ]
606.  
607. num_labels = len(labels)
608. num_columns = 3
609. rows_per_column = num_labels // num_columns + 1
610.  
611. # Create a dictionary to hold the entry fields
612. entry_fields = {}
613.  
614. default_values = {}
615. for i, label_text in enumerate(labels):
616.  
617.     label = tk.Label(input_frame, text=label_text + ":", font=("Arial", 10))
618.     label.grid(column=i % num_columns * 2, row=i // num_columns, sticky="e", pady=20)
619.     
620.     default_value = default_values.get(label_text,round(random.uniform(-5,5),2))
621.     entry = tk.Entry(input_frame, font=("Arial", 11))
622.     entry.insert(tk.END, default_value)
623.     entry.grid(column=i % num_columns * 2 + 1, row=i // num_columns, pady=20)
624.     
625.     # Store the entry field in the dictionary
626.     entry_fields[label_text] = entry
627.  
628.  
629. text_label = tk.Label(window, text="BREAST CANCER DIAGNOSIS SYSTEM",
630.                       bg="#D3D3D3", fg="black", font=("Arial", 16), padx=10, pady=10)
631. text_label.place(relx=0.5, rely=0.06, anchor=tk.CENTER)
632.  
633. # Create the predict button
634. predict_button = tk.Button(window, text="Submit", command=predict_button_click,
635.                            bg="#4caf50", fg="white", font=("Arial", 12), padx=10, pady=10)
636. predict_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
637.  
638.  
639. # Run the application
640. window.mainloop()
