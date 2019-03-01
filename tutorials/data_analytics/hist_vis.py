import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p = r'CICIDS2017.csv'
p1 = r'small_set.csv'
d = pd.read_csv(p, engine='python')
source_ip = d['Source IP']
s = d['Source IP'].value_counts().plot(kind='bar')

plt.xlabel('IP addresses')
plt.ylabel('frequency')
plt.title('histogram of source IP\'s')
plt.show()
