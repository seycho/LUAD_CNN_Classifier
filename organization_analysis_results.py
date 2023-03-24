import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import cv2, os


def MakeDirTree(pathDir, pathRoot=""):
    pathSplit = pathDir.split('/', 1)
    pathFull = os.path.join(pathRoot, pathSplit[0])
    if not os.path.isdir(pathFull):
        os.mkdir(pathFull)
    if (len(pathSplit[1]) != 0):
        MakeDirTree(pathSplit[1], pathRoot=pathFull)
    return None

def LoadImagePath(pathRoot, pathDir, condition):
    pathDic = {}
    for path0 in pathDir:
        for path1 in os.listdir(os.path.join(pathRoot, path0, condition)):
            pathImage = os.path.join(pathRoot, path0, condition, path1)
            _id = path1.split('.')[0]
            if _id not in pathDic:
                pathDic[_id] = []
            pathDic[_id].append(pathImage)
    return pathDic

def SaveImageGIF(pathList, savePath):
    frames = []
    for _idx, pathImage in enumerate(pathList):
        img = imageio.imread(pathImage)
        cv2.putText(img, "Cycle %d"%_idx, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0))
        frames.append(img)
    imageio.mimsave(savePath, frames, format='GIF', duration=1)
    return None

def main():

    print("Prepare")
    MakeDirTree("analysis/gif/overlap/")
    MakeDirTree("analysis/gif/compare/")

    overlap = {}
    compare = {}
    pathRoot = "analysis/"
    pathDir = [LD for LD in os.listdir(pathRoot) if "results" in LD]
    pathDir.sort()

    print("Merging image as GI")
    overlap = LoadImagePath(pathRoot, pathDir, "overlap")
    compare = LoadImagePath(pathRoot, pathDir, "compare")

    for _id in compare.keys():
        SaveImageGIF(overlap[_id], os.path.join("analysis/gif/overlap/", "%s.gif"%_id))
        SaveImageGIF(compare[_id], os.path.join("analysis/gif/compare/", "%s.gif"%_id))
    print("merged images are saved analysis/gif/")

    print("Draw graph")
    stat = {}
    for path0 in pathDir:
        variables = dict(np.loadtxt(os.path.join(pathRoot, path0, "statistical.txt"), delimiter="\t", dtype=str))
        for key, value in variables.items():
            if key not in stat:
                stat[key] = []
            stat[key].append(float(value))

    plt.figure(figsize=(len(stat)*5, 3))
    for _idx, key in enumerate(stat.keys()):
        plt.subplot(1, len(stat), _idx+1)
        plt.title(key)
        plt.plot(stat[key])
    plt.savefig("analysis/statistical_graph.jpg", bbox_inches='tight')
    plt.close()
    print("statistical graph are saved analysis/statistical_graph.jpg")

    return None

if __name__ == "__main__":
    main()