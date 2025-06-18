import matplotlib.pyplot as plt
import numpy as np
from mannrs import Tensor
from tqdm import tqdm

ae, L, gamma = 0.2, 30, 3.9
Lx, Ly, Lz = 6000, 200, 200
Nx = 1000
tol, min_depth = 0.5, 2
tensors = {
    "isotropic": Tensor.Isotropic(ae, L),
    "sheared": Tensor.Sheared(ae, L, gamma),
    "sheared_sinc": Tensor.ShearedSinc(ae, L, gamma, Ly, Lz, tol, min_depth),
}
if __name__ == "__main__":
    res = {}

    dx = Lx / (Nx - 1)
    kxs = np.linspace(-0.05, 0.05, 100)
    kys = np.linspace(-0.1, 0.1, 100)
    kx_grid, ky_grid = np.meshgrid(kxs, kys)
    for name, tensor in tensors.items():
        Suu = np.zeros((100, 100))
        print(name)
        for i, kx in tqdm(enumerate(kxs)):
            for j, ky in tqdm(enumerate(kys)):
                Suu[i, j] = tensor.tensor((kx, ky, 0.0))[0, 0]
        res[name] = Suu

    fig, axes = plt.subplots(1, 3)
    for ax, (name, Suu) in zip(axes, res.items()):
        # plt.plot(ks, Suu, label=name)
        ax.imshow(Suu)
        ax.set_title(name)

    plt.legend()
    plt.show()
