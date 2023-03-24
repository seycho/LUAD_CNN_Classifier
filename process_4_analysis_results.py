from modules import *

import matplotlib.pyplot as plt
import numpy as np
import openslide, argparse, pickle, os

from sklearn.metrics import roc_curve, auc


def MakeDirTree(pathDir, pathRoot=""):
    pathSplit = pathDir.split('/', 1)
    pathFull = os.path.join(pathRoot, pathSplit[0])
    if not os.path.isdir(pathFull):
        os.mkdir(pathFull)
    if (len(pathSplit[1]) != 0):
        MakeDirTree(pathSplit[1], pathRoot=pathFull)
    return None

def main():

    parser = argparse.ArgumentParser(description="Preprocess multi threading code.")
    parser.add_argument("--tsv", type=str, help="parameters recorded tsv text file path.")
    args = parser.parse_args()

    print("Analysis MIL results.")
    print("  Load variables")
    variables = dict(np.loadtxt(args.tsv, dtype=str, delimiter="\t"))
    sizeMicronXY = np.array(variables["sizeMicronXY"].split(','), dtype=int)
    intervalMicronXY = np.array(variables["intervalMicronXY"].split(','), dtype=int)
    resizePixelXY = np.array(variables["resizePixelXY"].split(','), dtype=int)
    saveAnalyPath = variables["saveAnalyPath"]
    loadInfosPath = variables["loadInfosPath"]
    loadLabelPath = variables["loadLabelPath"]

    print("  Load results.")
    infoPack = pickle.load(open(loadInfosPath, "rb"))
    correctMILDic = pickle.load(open(loadLabelPath, "rb"))

    print("  Folder tree check.")
    MakeDirTree(os.path.join(saveAnalyPath, "confusion_matrix/"))
    MakeDirTree(os.path.join(saveAnalyPath, "compare/"))
    MakeDirTree(os.path.join(saveAnalyPath, "overlap/"))

    print("  Set WSI class.")
    cur = LoginWSIViewer()
    wsiluadinfo = GetWSIInfos(cur, "wsiluadinfo")

    maskGTDic = {}
    maskAIDic = {}
    intervalPixelXYDic = {}
    patchImporterDic = {}
    for WSIID in wsiluadinfo.keys():
        patchImporterDic[WSIID] = WSIPatchImporter(wsiluadinfo[WSIID]["filepath"], wsiluadinfo[WSIID]["maskspec"], wsiluadinfo[WSIID]["maskanno"])
        patchImporterDic[WSIID].MakePatchCoordinates(sizeMicronXY, intervalMicronXY, resizePixelXY)
        intervalPixelXYDic[WSIID] = patchImporterDic[WSIID].intervalPixelXY
        maskGTDic[WSIID] = np.zeros(patchImporterDic[WSIID].numPatch[::-1])
        maskAIDic[WSIID] = np.zeros(patchImporterDic[WSIID].numPatch[::-1])

    print("  Make label mask.")
    for _code, _label in zip(infoPack["code"], infoPack["label"]):
        WSIID, coordinate = _code.split("\t")
        coordinate = np.array(coordinate.split(','), dtype=int)
        x, y = (coordinate / intervalPixelXYDic[WSIID]).astype(int)[::-1]
        maskGTDic[WSIID][x, y] = _label[1]

    for _code, _label in list(correctMILDic.items()):
        WSIID, coordinate = _code.split("\t")
        coordinate = np.array(coordinate.split(','), dtype=int)
        x, y = (coordinate / intervalPixelXYDic[WSIID]).astype(int)[::-1]
        maskAIDic[WSIID][x, y] = _label[1]

    print("  Draw compare images.")
    for WSIID in wsiluadinfo.keys():
        plt.figure(figsize=np.array(DiagnoalNormalize(maskAIDic[WSIID].shape[1]*3, maskAIDic[WSIID].shape[0]))*15)
        level = patchImporterDic[WSIID].handle["WSI"].level_count - 1
        boundsDic = patchImporterDic[WSIID].boundsDic
        _w, _h = (np.array([boundsDic['w'], boundsDic['h']]) / patchImporterDic[WSIID].downsampleDic["WSI"][level]).round().astype(int)
        maskSum = maskGTDic[WSIID].round() + maskAIDic[WSIID].round()
        plt.suptitle("IoU = %.2f%%"%((maskSum == 2).sum() / (maskSum >= 1).sum()*100))
        plt.subplot(1,3,1)
        plt.imshow(patchImporterDic[WSIID].LoadImage([0, 0], level, [_w, _h], [_w, _h]))
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.imshow(maskGTDic[WSIID], vmin=0, vmax=1)
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.imshow(maskAIDic[WSIID], vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig(os.path.join(saveAnalyPath, "compare/%s.png"%(WSIID)))
        plt.clf()
        plt.close()
        img = patchImporterDic[WSIID].LoadImage([0, 0], level, [_w, _h], [_w, _h])
        ShowOverlapMask(img, maskAIDic[WSIID], pathSave=os.path.join(saveAnalyPath, "overlap/%s.png"%(WSIID)))
    plt.clf()

    print("  Draw ROC curve.")
    labelGT = infoPack["label"][infoPack["code"].argsort()][:,1]
    labelAI = np.array(list(correctMILDic.values()))[np.array(list(correctMILDic.keys())).argsort()][:,1]
    x, y, _ = roc_curve(labelGT, labelAI, pos_label=1)
    plt.plot(x, y)
    plt.title("AUC = %.2f%%"%(auc(x, y)*100))
    plt.savefig(os.path.join(saveAnalyPath, "roc.png"))
    plt.clf()

    print("  Draw patch image confusion matrix.")
    codeConfMatxDic = {"TP" : [],"FN" : [],"FP" : [],"TN" : []}
    labelConfMatxDic = {"TP" : [],"FN" : [],"FP" : [],"TN" : []}
    for _code, _label in zip(infoPack["code"], infoPack["label"]):
        _GT = _label.argmax()
        _AI = correctMILDic[_code].argmax()
        _type = ""
        if (_GT == _AI):
            _type += 'T'
        else:
            _type += 'F'
        if (_GT == 1):
            _type += 'P'
        else:
            _type += 'N'
        codeConfMatxDic[_type].append(_code)
        labelConfMatxDic[_type].append(correctMILDic[_code])

    for _key in codeConfMatxDic.keys():
        idxList = np.array(labelConfMatxDic[_key]).max(1).argsort()[-64:]
        plt.figure(figsize=(9, 9))
        plt.suptitle(_key)
        for _idx in range(64):
            plt.subplot(8, 8, _idx+1)
            WSIID, coordiante = codeConfMatxDic[_key][idxList[_idx]].split("\t")
            coordinate = np.array(coordiante.split(','), dtype=int)
            plt.imshow(patchImporterDic[WSIID].LoadImage(coordinate))
            plt.axis("off")
        plt.savefig(os.path.join(saveAnalyPath, "confusion_matrix/patch_image_%s.png"%(_key)))
        plt.clf()

    print("  Calculate statistical values.")
    cmArray = np.array([[len(codeConfMatxDic["TP"]), len(codeConfMatxDic["FP"])],
                        [len(codeConfMatxDic["FN"]), len(codeConfMatxDic["TN"])]])

    accuracy = (cmArray[0][0]+cmArray[1][1]) / (cmArray.sum())
    sensitivity = cmArray[0][0] / cmArray[:,0].sum()
    specificity = cmArray[1][1] / cmArray[:,1].sum()
    PPV = cmArray[0][0] / cmArray[0].sum()
    NPV = cmArray[1][1] / cmArray[1].sum()

    record = np.array([["accuracy", str(accuracy)],
                       ["sensitivity", str(sensitivity)],
                       ["specificity", str(specificity)],
                       ["PPV", str(PPV)],
                       ["NPV", str(NPV)],
                      ])
    np.savetxt(os.path.join(saveAnalyPath, "statistical.txt"), record, fmt="%s", delimiter="\t")
    print("  Analysis datas are saved %s."%saveAnalyPath)
    print()

    return None


if __name__ == "__main__":
    main()