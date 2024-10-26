import pygame
import json
import cv2
import numpy as np
from os import listdir
from pathlib import Path
pygame.init()


def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    rangec = (maxc-minc)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = rangec / maxc
    rc = (maxc-r) / rangec
    gc = (maxc-g) / rangec
    bc = (maxc-b) / rangec
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) # % 1.0
    return h, s, v


base = Path("tests/resources/kfline")
files = [f for f in listdir(base) if f.endswith((".png", ".jpg"))]
files = [f for f in files if f.startswith("Chiika_Fujiwara X Kooli.MIR")]

images = [pygame.image.load(base / file) for file in files]

data = {}
i = -1

def save():
    with open(base / "colours-single.json", "w") as f_out:
        json.dump(data, f_out, indent=2)


def next():
    global i, window
    i += 1
    save()
    if i >= len(images):
        exit()

    image = images[i]
    window = pygame.display.set_mode(image.get_size())


colour = [0.0, 0.0]
cn = 0
c1, c2 = None, None

next()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                c1 = [colour[0]/cn, colour[1]/cn]
                colour = [0., 0]
                cn = 0
            if event.key == pygame.K_2:
                c2 = [colour[0]/cn, colour[1]/cn]
                colour = [0., 0]
                cn = 0

            if event.key == pygame.K_RETURN:
                file = files[i]
                data[Path(file).stem] = [c1, c2]
                next()
    
    image = images[i]
    window.blit(image, (0, 0))

    if pygame.mouse.get_pressed()[0]:
        pos = pygame.mouse.get_pos()
        col = image.get_at(pos)
        np_col = np.array([[[col.r, col.g, col.b]]], dtype=np.uint8)

        lab = cv2.cvtColor(np_col, cv2.COLOR_RGB2LAB)
        print(lab)
        # colour[0] += hsv[0]
        # colour[1] += hsv[1]
        cn += 1

    pygame.display.update()