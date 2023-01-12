from PIL import Image
from PIL.ExifTags import TAGS

import sys
if len (sys.argv) != 2 :
    print("Usage: python image_metadata.py imagefile.png ")
    sys.exit (1)

image_file = sys.argv[1]
# open the image file
image = Image.open(image_file)
lowercase_filename = image_file.lower()

if lowercase_filename.endswith('.png'):
    #print(image.text) 
    for k, v in image.text.items():
        print(k,": ", v)
elif lowercase_filename.endswith('.jpg'):
# extracting the exif metadata
    exifdata = image.getexif() 
    # looping through all the tags present in exifdata
    for tagid in exifdata:
        # getting the tag name instead of tag id
        tagname = TAGS.get(tagid, tagid)
        # passing the tagid to get its respective value
        value = exifdata.get(tagid)
        # printing the final result
        print(f"{tagname:25}: {value}")
else:
    print("file type not supported")



