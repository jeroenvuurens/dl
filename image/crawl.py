from icrawler.builtin import GoogleImageCrawler
import ipywidgets as widgets
from tqdm import tqdm_notebook as tqdm
from imagecleaner.cli import *
import os

def crawl_images(folder, query, threads = 1, max_num=100, category=None, size = 'medium', type='photo', min_size=None, max_size=None, offset = 0, file_idx_offset = 0):
    category = category if category else query.split()[0]
    folder = str(folder) + '/' + category
    filters = dict(size=size, type=type)
    GoogleImageCrawler(storage={'root_dir': folder}, downloader_threads = threads).crawl(keyword=query + ' jpg', filters=filters, max_num = max_num, offset=offset, min_size=min_size, max_size=max_size, file_idx_offset = file_idx_offset )

def image_filter(path, category, columns=4, height=200, width=200):
    def on_click(button):
        for r in rows:
            if type(r) is widgets.HBox:
                for c in r.children:
                    checkbox = c.children[1]
                    if checkbox.value:
                        print(checkbox.description_tooltip)
                        os.remove(checkbox.description_tooltip)
                        
    imagefiles = [f for f in path.glob(category + '/*')]
    rows = []
    cols = []
    for i, imgfile in enumerate(tqdm(imagefiles)):
        row = i // columns
        col = i % columns
        img = open(imgfile, 'rb').read()
        image = widgets.Image( value=img, width=width, height=height )
        button = widgets.Checkbox( description='Delete', description_tooltip = str(imgfile) )
        box = widgets.VBox([image, button])
        cols.append(box)
        if len(cols) == columns:
            rows.append(widgets.HBox(cols))
            cols = []
    if len(cols) > 0:
        rows.append(widgets.HBox(cols))
    button = widgets.Button( description='Delete' )
    button.on_click(on_click)
    rows.append(button)
    return widgets.VBox(rows)

def image_remove_duplicates(path):
    remove_images(str(path), 16, 6)
