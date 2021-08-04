import os
from PIL import Image


if __name__ == '__main__':
    dir_name = 'log/sliding_lid_W0.10_visc0.03_size300x300/fig/'
    fs = [dir_name + f for f in os.listdir(dir_name) if f.endswith('.png')]
    fs = sorted(fs)
    images = list(map(lambda file: Image.open(file), fs))
    images.pop(0).save('lid-driven-cavity-w0.1-visc0.03-small.gif', save_all=True, append_images=images, duration=50, optimize=False, loop=0)
