# Forecasting-of-Maruti-Suzuki-Inventory-Stocks
Internship Project

Opex is a consultancy firm and Maruti Suzuki was their client.I was assigned to the peoject of Maruti Suzuki.The problem of the client was  that they have a sequence(time series) dataset of previous sell record of Each part type associated with a car,like Wiper,SeatCovers etc.which they labelled those as SKUIDs’ in different locations in India(Pune,Mumbai,Kolkata,Delhi etc).They problem statement was to forecast the SKUID demands using advance ML techniques so that they don’t run out of stocks which they suffered in the Past.Their past base model was giving a forecasting accuracy 81.34.My task was to improve it.

Detecting Anomalies: The dataset was full of anomalies.So first just read some research papers what are the new techniques available for anomaly detection and found OneSVM,Autoencoder.Finally reached in a conculsion that in my model will use Autoencoder.

Models Used:Used Deep Learning model Autoencoder LSTM as the benchmark model.

Inference:Was able to reach an accuracy of 88.71 in the test set.

Used language and Frameworks:Used Python(IDE:Pycharm) as programming language and used deep learning frameworks
Tensorflow and Keras to do that.

Visualization:All the visualization was made on Tableau.
