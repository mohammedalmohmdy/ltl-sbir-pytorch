
# Examples & Templates

Below are CSV templates for dataset metadata.

## `meta/images.csv`
```
path,label
images/0001.jpg,1001
images/0002.jpg,1002
images/0003.jpg,1003
```

## `meta/sketches.csv`
```
path,label
sketches/0001.png,1001
sketches/another_style_0001.png,1001
sketches/0002.png,1002
```

## Optional: `meta/train_pairs.csv`
```
sketch_path,pos_img_path,neg_img_path
sketches/0001.png,images/0001.jpg,images/0002.jpg
sketches/0002.png,images/0002.jpg,images/0003.jpg
```
