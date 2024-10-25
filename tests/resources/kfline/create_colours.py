import pygame
import json
from os import listdir
from pathlib import Path
from colorsys import rgb_to_hsv
pygame.init()


base = Path("tests/resources/kfline")
files = [f for f in listdir(base) if f.endswith((".png", ".jpg"))]
images = [pygame.image.load(base / file) for file in files]


data = {}
i = 1

def save():
    with open(base / "colours2.json", "w") as f_out:
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
        hsv = rgb_to_hsv(col.r, col.g, col.b)
        colour[0] += hsv[0]
        colour[1] += hsv[1]
        cn += 1

    pygame.display.update()