from modules import *

import numpy as np
import pickle, argparse, os

from multiprocessing import Pool


def MakePatchInfos(inputList):

    WSIID = inputList[0]
    pathWSI = inputList[1]["pathWSI"]
    pathSpec = inputList[1]["pathSpec"]
    pathAnno = inputList[1]["pathAnno"]
    sizeMicronXY = inputList[2]["sizeMicronXY"]
    intervalMicronXY = inputList[2]["intervalMicronXY"]
    resizePixelXY = inputList[2]["resizePixelXY"]

    patchImporter = WSIPatchImporter(pathWSI, pathSpec, pathAnno)
    coordinatesPatch = patchImporter.MakePatchCoordinates(sizeMicronXY, intervalMicronXY, resizePixelXY)

    patchIsUseful = patchImporter.IsUsfulMask("spec", coordinatesPatch, ratioPass=0.7)
    patchIsPositive = patchImporter.IsUsfulMask("anno", coordinatesPatch, ratioPass=0.5)

    coordinatesPatchDic = {}
    coordinatesPatchDic["pos"] = coordinatesPatch[patchIsUseful * patchIsPositive]
    coordinatesPatchDic["neg"] = coordinatesPatch[patchIsUseful * (patchIsPositive==False)]

    patchInfosDic = {}
    for _type in ["pos", "neg"]:
        for coordinate in coordinatesPatchDic[_type]:
            if _type == "pos":
                patchInfosDic["%s\t%d,%d"%(WSIID,coordinate[0],coordinate[1])] = [0, 1]
            else:
                patchInfosDic["%s\t%d,%d"%(WSIID,coordinate[0],coordinate[1])] = [1, 0]

    return patchInfosDic

def main():

    parser = argparse.ArgumentParser(description="Preprocess multi threading code.")
    parser.add_argument("--tsv", type=str, help="parameters recorded tsv text file path.")
    args = parser.parse_args()

    print("Prepare CNN model train dataset.")
    print("  Load variables")
    variables = dict(np.loadtxt(args.tsv, dtype=str, delimiter="\t"))
    sizeMicronXY = np.array(variables["sizeMicronXY"].split(','), dtype=int)
    intervalMicronXY = np.array(variables["intervalMicronXY"].split(','), dtype=int)
    resizePixelXY = np.array(variables["resizePixelXY"].split(','), dtype=int)
    numThread = int(variables["numThread"])
    saveInfosPath = variables["saveInfosPath"]
    loadInfosPath = variables["loadInfosPath"]
    refLabelPath = variables["refLabelPath"]
    del variables, args

    if os.path.isfile(loadInfosPath):
        print("  Load patch image dataset.")
        infoPack = pickle.load(open(loadInfosPath, "rb"))
        packCode = infoPack["code"]
        packLabel = infoPack["label"].astype(float)
    else:
        print("  Make new patch image dataset.")
        print("    Set WSI class.")
        cur = LoginWSIViewer()
        wsiluadinfo = GetWSIInfos(cur, "wsiluadinfo")

        patchImporterDic = {}
        for WSIID in wsiluadinfo.keys():
            patchImporterDic[WSIID] = WSIPatchImporter(wsiluadinfo[WSIID]["filepath"],
                                                       wsiluadinfo[WSIID]["maskspec"],
                                                       wsiluadinfo[WSIID]["maskanno"])

        sizePack = {"sizeMicronXY" : sizeMicronXY,
                    "intervalMicronXY" : intervalMicronXY,
                    "resizePixelXY" : resizePixelXY}

        print("    Prepare multi-threading data.")
        inputList = []
        for WSIID in wsiluadinfo.keys():
            pathPack = {"pathWSI" : wsiluadinfo[WSIID]["filepath"],
                        "pathSpec" : wsiluadinfo[WSIID]["maskspec"],
                        "pathAnno" : wsiluadinfo[WSIID]["maskanno"]}
            inputList.append([WSIID, pathPack, sizePack])

        print("    Multi-threading process.")
        with Pool(numThread) as p:
            poolOutput = p.map(MakePatchInfos, inputList)

        patchInfosDic = {}
        for patchInfosDicSub in poolOutput:
            patchInfosDic.update(patchInfosDicSub)

        packCode = np.array(list(patchInfosDic.keys()))
        packLabel = np.array(list(patchInfosDic.values()), dtype=float)

    if os.path.isfile(refLabelPath):
        print("  Load reference MIL results.")
        labelDicMIL = pickle.load(open(refLabelPath, "rb"))

        print("    Remaking label.")
        for _idx in range(len(packCode)):
            if packCode[_idx] in labelDicMIL:
                packLabel[_idx] = (packLabel[_idx]*0.3 + labelDicMIL[packCode[_idx]]*0.7).round(2)

    infoPack = {"code" : packCode, "label" : packLabel}
    pickle.dump(infoPack, open(saveInfosPath, "wb"))
    print("  Patch image data file is saved %s."%saveInfosPath)

    return None

if __name__ == "__main__":
    main()
