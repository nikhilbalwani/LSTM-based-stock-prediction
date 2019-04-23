import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Data_Normalization import generate_data

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')

read_data = pd.read_csv('sp500.csv')
#read_data = read_data[['Close']]

x, y, x_test, y_test = generate_data(read_data, feature_cols=['Close', 'Volume'], test_ratio=0.15)
print()
data = [y_test[i][0] for i in range(y_test.shape[0])]
#print(data)

N = 50
alpha = 2 / (N + 1)	# For 50 days EWMA
ewm = 0

# plot
plt.figure(figsize=(16,8))
plt.plot(data[0:600])

predicted_out = []

start_time = time.time()

for j in range(0,550,50):
	
	#splitting into train and validation
	train = data[j:j+50]
	valid = data[j+50:j+100]
	#ewm = 0
	
	#make predictions
	preds = []
	for i in range(0,len(valid)):
		ewm = (1 - alpha) * ewm + alpha * train[i]
		#ewm = ewm / (1 - alpha ** i)
		preds.append(ewm)
		
	#calculate rmse
	rms = np.sqrt(np.mean(np.power((np.array(valid) - preds), 2)))
	print("RMS: ", rms)

	#valid['Predictions'] = 0
	#valid['Predictions'] = preds
	#plt.plot(train['Close'])
	#plt.plot(valid[['Close', 'Predictions']])
	#plt.plot(valid['Predictions'])
	predicted_out += preds
	
end_time = time.time()


ax = fig.add_subplot(111)
ax.plot(y_test, label='True Data')
plt.legend()

index = [i for i in range(50, 600)]
plt.plot(index, predicted_out)

plt.savefig('expo.pdf')

print("\nAverage time per loop: {}".format((end_time - start_time)/11))

plt.show()
