import os, shutil
savedShapes = '.\savedShapes'
for shapeFolder in os.listdir(savedShapes):
    file_path = os.path.join(savedShapes, shapeFolder)
    try:
        for image in os.listdir(file_path):
            os.remove(file_path + "\\" + image)

    except Exception as e:
        print(e)
