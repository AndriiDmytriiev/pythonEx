import numpy as nump
import pandas as pand
import matplotlib.pyplot as plt_ex
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest

# Data creation: simulate growth with noise
nump.random.seed(42)
time = nump.arange(0, 100)
values = 50 + 0.5 * time + nump.random.normal(0, 5, size=time.shape[0])

# Let's add anomalies
values[[10, 50, 90]] = [100, 10, 120]

# Let's convert to DataFrame
data = pand.DataFrame({'Time': time, 'Values': values})
slp, itcept, r_value, p_value, std_err = linregress(data['Time'], data['Values'])

# Trend formula
data['Trend'] = slp * data['Time'] + itcept






print(f"Trend line: y = {slp:.2f}x + {itcept:.2f}")
plt_ex.figure(figsize=(10, 6))
plt_ex.plot(data['Time'], data['Values'], label='Data', color='blue', marker='o')
plt_ex.plot(data['Time'], data['Trend'], label='Trend', color='red', linestyle='--')
plt_ex.title('Trending data analysis')
plt_ex.xlabel('Time')
plt_ex.ylabel('Value')
plt_ex.legend()
plt_ex.grid(True)
plt_ex.show()
# Anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = model.fit_predict(data[['Values']])

# Let's denote anomalies
anomalies = data[data['Anomaly'] == -1]

# Visualization
plt_ex.figure(figsize=(10, 6))
plt_ex.plot(data['Time'], data['Values'], label='Data', color='blue', marker='o')
plt_ex.scatter(anomalies['Time'], anomalies['Values'], color='red', label='Anomalies', s=100, edgecolors='black')
plt_ex.title('Detecting anomalies in data')
plt_ex.xlabel('Time')
plt_ex.ylabel('Value')
plt_ex.legend()
plt_ex.grid(True)
plt_ex.show()