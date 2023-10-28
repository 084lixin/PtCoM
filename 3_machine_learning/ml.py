import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
import random
import pickle
from sklearn.pipeline import Pipeline

# Load feature and energy data
x = np.loadtxt("../2_feature/feature.csv", delimiter=",",encoding='utf-8-sig')
y = np.loadtxt("../2_feature/energy.csv", delimiter=",",encoding='utf-8-sig')

# Define data preprocessing and Gaussian Process Regressor
standardscaler = StandardScaler()
pca = PCA(0.99)
kernel = gp.kernels.ConstantKernel(constant_value=1,
                                   constant_value_bounds=(1e-5,1e3)) *gp.kernels.RBF(length_scale=10, length_scale_bounds=(1e-5, 1e3))
gpr_model = gp.GaussianProcessRegressor(kernel=kernel,
                                        n_restarts_optimizer=50,
                                        alpha=0.1,
                                        normalize_y=True)

# Lists to store evaluation metrics
MAE_TRAIN = []
R2_TRAIN = []
MAE_TEST = []
R2_TEST = []

# Train and evaluate the model multiple times (random_state 0 to 9)
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8,test_size=0.2, random_state=i)
    standardscaler = StandardScaler()
    pca = PCA(0.99)
    X_train = standardscaler.fit_transform(X_train)
    X_train = pca.fit_transform(X_train)
    X_test = standardscaler.transform(X_test)
    X_test = pca.transform(X_test)  
    gpr_model.fit(X_train, y_train)

    y_train_predict=gpr_model.predict(X_train)
    y_test_predict=gpr_model.predict(X_test)

    MAE_train = mean_absolute_error(y_pred=y_train_predict, y_true=y_train)
    R2_train = r2_score(y_pred=y_train_predict, y_true=y_train)
    MAE_test = mean_absolute_error(y_pred=y_test_predict, y_true=y_test)
    R2_test = r2_score(y_pred=y_test_predict, y_true=y_test)

    MAE_TRAIN.append(MAE_train)
    MAE_TEST.append(MAE_test)
    R2_TRAIN.append(R2_train)
    R2_TEST.append(R2_test)

# Print average metrics for training and testing    
print('MAE train:',np.mean(MAE_TRAIN),'R2 train:',np.mean(R2_TRAIN))
print('MAE test:',np.mean(MAE_TEST),'R2 test:',np.mean(R2_TEST))

# Re-train the model with a specific random state (random_state=10)
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=10)
standardscaler = StandardScaler()
pca = PCA(0.99)
X_train = standardscaler.fit_transform(X_train)
X_train = pca.fit_transform(X_train)
X_test = standardscaler.transform(X_test)
X_test = pca.transform(X_test)  
gpr_model.fit(X_train, y_train)

# Predict and evaluate with a specific random state
y_train_predict=gpr_model.predict(X_train)
y_test_predict=gpr_model.predict(X_test)

MAE_train = mean_absolute_error(y_pred=y_train_predict, y_true=y_train)
R2_train = r2_score(y_pred=y_train_predict, y_true=y_train)
MAE_test = mean_absolute_error(y_pred=y_test_predict, y_true=y_test)
R2_test = r2_score(y_pred=y_test_predict, y_true=y_test)

print('MAE train:',MAE_train,'R2 train:',R2_train)
print('MAE test:',MAE_test,'R2 test:',R2_test)

# Create a parity plot
plt.rc('font', family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
xmin = min(y) - 2
xmax = max(y) + 2
x_lim = np.linspace(xmin, xmax, 20)

fig, ax = plt.subplots(figsize=(6, 6))
plt.xlim([xmin, xmax])
plt.ylim([xmin, xmax])
plt.scatter(y_train_predict, y_train, color='darkcyan')
plt.scatter(y_test_predict, y_test, color='deeppink')
plt.legend(['train', 'test'], fontsize=14)
plt.plot(x_lim, x_lim, linestyle='--', color='dimgray')

plt.xlabel('GPR energies (eV)', fontsize=18)
plt.ylabel('DFT energies (E)', fontsize=18)
plt.savefig('parity plot.png', dpi=1000)

# Save the GPR model and preprocessing pipeline
s = pickle.dumps(gpr_model)
with open('GPRmodel.model', 'wb+') as f:
    f.write(s)

steps = [('scaler', standardscaler), ('pca', pca)]
pipeline = Pipeline(steps)
with open('Preprocessing.pkl', 'wb+') as f:
    pickle.dump(pipeline, f)