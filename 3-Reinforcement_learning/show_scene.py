import numpy as np
import matplotlib.pyplot as plt

model = np.load('model.npy')
tmp = np.zeros((48, 4, 2))
tmp[0, :, :] = model[0, :, :]
tmp[11, :, :] = model[1, :, :]
tmp[12:, :, :] = model[2:, :, :]
model = tmp

fig, ax = plt.subplots(nrows=4, ncols=2)

ax[0, 0].set_title('R, right')
ax[1, 0].set_title('R, down')
ax[2, 0].set_title('R, left')
ax[3, 0].set_title('R, up')
ax[0, 1].set_title('S, right')
ax[1, 1].set_title('S, down')
ax[2, 1].set_title('S, left')
ax[3, 1].set_title('S, up')

for k in range(4):

	ax[k, 0].matshow(np.flip(model[:, k, 0].reshape(4, 12), axis=0))
	ax[k, 1].matshow(np.flip(model[:, k, 1].reshape(4, 12), axis=0))

	for (i, j), z in np.ndenumerate(np.flip(model[:, k, 0].reshape((4, 12)), axis=0)):
		ax[k, 0].text(j, i, '{:1.0f}'.format(z), ha='center', va='center')
	for (i, j), z in np.ndenumerate(np.flip(model[:, k, 1].reshape((4, 12)), axis=0)):
		ax[k, 1].text(j, i, '{:1.0f}'.format(z), ha='center', va='center')


plt.tight_layout()
plt.show()
