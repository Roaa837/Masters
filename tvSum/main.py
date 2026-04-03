import h5py

data = h5py.File("ydata-tvsum50.mat", "r")

dataset = data['tvsum50']
print(dataset.keys())