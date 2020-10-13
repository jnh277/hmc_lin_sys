import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_dbode_ML
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
import pystan
from scipy.io import savemat



recompile = True

# specific data path

data_path = 'data/stochastic_volatility_data1.mat'


data = loadmat(data_path)

y = np.squeeze(data['y'])
x_true = np.squeeze(data['xx'])
a_true = data['a'][0,0]
b_true = data['b'][0,0]
c_true = data['c'][0,0]
muP = data['mup'][0,0]      # prior initial state mean
cP = data['cP'][0,0]        #prior initial state variance


model_path = 'stan/stochastic_volatility.pkl'
if Path(model_path).is_file() and not recompile:
    model = pickle.load(open(model_path, 'rb'))
else:
    model = pystan.StanModel(file='stan/stochastic_volatility.stan')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


stan_data = {
    'N':len(y),
    'y':y,
    'muP':muP,
    'cP':cP
}

iter = 8000
fit = model.sampling(data=stan_data,thin=3, iter=iter, chains=4)

# extract the results
traces = fit.extract()

a_traces = traces['a']
b_traces = traces['b']
c_traces = traces['c']

x_traces = traces['z']

lp_traces = traces['lpd']

plt.hist(lp_traces,bins=30,density=True)
plt.axvline(lp_traces.mean(),linestyle='--',color='r',linewidth=3.5)
sns.kdeplot(lp_traces,linewidth=2)
plt.legend(['mean log p(y)','sample log p(y)'])
plt.xlabel('log p(y)')
plt.show()

blims = (b_traces.min(),b_traces.max())
alims = (a_traces.min(),a_traces.max())

plt.subplot(3,3,1)
plt.hist(a_traces,30,density=True)
plt.axvline(a_true,linestyle='--',color='r',linewidth=3.5)
plt.ylabel('a')
plt.xlim(alims)

plt.subplot(3,3,4)
# plt.scatter(a_traces,b_traces)
# plt.hist2d(a_traces,b_traces,bins=30)
tmp = np.concatenate((np.reshape(a_traces,(-1,1)),np.reshape(b_traces,(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.ylim(blims)
plt.xlim(alims)
plt.ylabel('b')

plt.subplot(3,3,5)
plt.hist(b_traces,30,density=True)
plt.axvline(b_true,linestyle='--',color='r',linewidth=3.5)
plt.xlim(blims)

# c_var_traces = c_traces**2
c_var_traces = c_traces
clims = (np.log(np.sqrt(c_var_traces)).min(),np.log(np.sqrt(c_var_traces)).max())

plt.subplot(3,3,7)
# plt.scatter(a_traces,np.log(np.sqrt(c_var_traces)))
tmp = np.concatenate((np.reshape(a_traces,(-1,1)),np.reshape(np.log(np.sqrt(c_var_traces)),(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlabel('a')
plt.ylabel('log(sqrt(c))')
plt.xlim(alims)
plt.ylim(clims)

plt.subplot(3,3,8)
# plt.scatter(b_traces,np.log(np.sqrt(c_var_traces)))
tmp = np.concatenate((np.reshape(b_traces,(-1,1)),np.reshape(np.log(np.sqrt(c_var_traces)),(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlabel('b')
plt.xlim(blims)
plt.ylim(clims)

plt.subplot(3,3,9)
plt.hist(np.log(np.sqrt(c_var_traces)),30,density=True)
plt.axvline(np.log(np.sqrt(c_true)),linestyle='--',color='r',linewidth=3.5)
plt.xlabel('log(sqrt(c))')
plt.xlim(clims)

plt.show()

plt.subplot(1,3,1)
plt.hist(x_traces[:,29],30,density=True)
plt.axvline(x_true[29],linestyle='--',color='r',linewidth=3.5)
plt.xlabel('x at t = 30')

plt.subplot(1,3,2)
plt.hist(x_traces[:,299],30,density=True)
plt.axvline(x_true[299],linestyle='--',color='r',linewidth=3.5)
plt.xlabel('x at t = 300')

plt.subplot(1,3,3)
plt.hist(x_traces[:,699],30,density=True)
plt.axvline(x_true[699],linestyle='--',color='r',linewidth=3.5)
plt.xlabel('x at t = 700')

plt.show()

plt.plot(np.mean(x_traces,0))
plt.plot(x_true)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a_traces, b_traces, np.log(np.sqrt(c_var_traces)))
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('log(sqrt(c))')
plt.show()

plt.subplot(3,3,1)
# plt.scatter(x_traces[:,29],a_traces)
tmp = np.concatenate((np.reshape(x_traces[:,29],(-1,1)),np.reshape(a_traces,(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,29].min(),x_traces[:,29].max()))
plt.ylabel(alims)
plt.xlabel('x at time = 30')
plt.ylabel('a')


plt.subplot(3,3,2)
tmp = np.concatenate((np.reshape(x_traces[:,29],(-1,1)),np.reshape(b_traces,(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,29].min(),x_traces[:,29].max()))
plt.ylabel(blims)
plt.xlabel('x at time = 30')
plt.ylabel('b')

plt.subplot(3,3,3)
# plt.scatter(x_traces[:,29],b_traces)
tmp = np.concatenate((np.reshape(x_traces[:,29],(-1,1)),np.reshape(np.log(np.sqrt(c_var_traces)),(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,29].min(),x_traces[:,29].max()))
plt.ylabel(clims)
plt.xlabel('x at time = 30')
plt.ylabel('log(sqrt(c))')

plt.subplot(3,3,4)
# plt.scatter(x_traces[:,29],a_traces)
tmp = np.concatenate((np.reshape(x_traces[:,299],(-1,1)),np.reshape(a_traces,(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,299].min(),x_traces[:,299].max()))
plt.ylabel(alims)
plt.xlabel('x at time = 300')
plt.ylabel('a')


plt.subplot(3,3,5)
tmp = np.concatenate((np.reshape(x_traces[:,299],(-1,1)),np.reshape(b_traces,(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,299].min(),x_traces[:,299].max()))
plt.ylabel(blims)
plt.xlabel('x at time = 300')
plt.ylabel('b')

plt.subplot(3,3,6)
# plt.scatter(x_traces[:,29],b_traces)
tmp = np.concatenate((np.reshape(x_traces[:,299],(-1,1)),np.reshape(np.log(np.sqrt(c_var_traces)),(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,299].min(),x_traces[:,299].max()))
plt.ylabel(clims)
plt.xlabel('x at time = 300')
plt.ylabel('log(sqrt(c))')

plt.subplot(3,3,7)
# plt.scatter(x_traces[:,29],a_traces)
tmp = np.concatenate((np.reshape(x_traces[:,699],(-1,1)),np.reshape(a_traces,(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,699].min(),x_traces[:,699].max()))
plt.ylabel(alims)
plt.xlabel('x at time = 700')
plt.ylabel('a')


plt.subplot(3,3,8)
tmp = np.concatenate((np.reshape(x_traces[:,699],(-1,1)),np.reshape(b_traces,(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,699].min(),x_traces[:,699].max()))
plt.ylabel(blims)
plt.xlabel('x at time = 700')
plt.ylabel('b')

plt.subplot(3,3,9)
# plt.scatter(x_traces[:,29],b_traces)
tmp = np.concatenate((np.reshape(x_traces[:,699],(-1,1)),np.reshape(np.log(np.sqrt(c_var_traces)),(-1,1))),1)
ax = sns.kdeplot(tmp, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
plt.xlim((x_traces[:,699].min(),x_traces[:,699].max()))
plt.ylabel(clims)
plt.xlabel('x at time = 700')
plt.ylabel('log(sqrt(c))')


plt.show()

savemat('results/stochastic_volatility_traces.mat',traces)