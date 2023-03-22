from modules import *

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision, torch, argparse, pickle, copy, os
import numpy as np


class DatasetClassWSI(Dataset):

    def __init__(self, patchImporterDic, patchInfos, transform):
        self.patchImporterDic = patchImporterDic
        self.codePack = patchInfos["code"]
        self.labelPack = patchInfos["label"]
        self.transform = transform

    def __getitem__(self, index):
        _code = self.codePack[index]
        label = self.labelPack[index]
        WSIID, coordinate = _code.split("\t")
        coordinate = np.array(coordinate.split(','), dtype=int)
        data = self.patchImporterDic[WSIID].LoadImage(coordinate)

        return _code, self.transform(Image.fromarray(data)), label

    def __len__(self):
        return len(self.codePack)

def TrainModel(model, dataLoader, criterion, optimizer, device, num):
    total = len(dataLoader.dataset)
    currentNum = 0
    correctSum = 0

    for _code, data, label in dataLoader:
        results = model(data.to(device))

        optimizer.zero_grad()
        loss = criterion(results, label.float().to(device))

        loss.backward()
        optimizer.step()

        currentNum += len(_code)
        correctSum += sum(results.cpu().max(1)[1] == label.max(1)[1]).item()

        print("Cycle %d | %d/%d [%.2f%%] | Acc = %.2f%%"%(num, currentNum, total, 100*currentNum/total, 100*correctSum/currentNum), end="\r")
    print()
    return None

def MakeInstance(model, classifier, dataLoader, device):
    total = len(dataLoader.dataset)
    currentNum = 0
    instanceCNN = {"pos" : {"code" : [], "result" : [], "instance" : []}, "neg" : {"code" : [], "result" : [], "instance" : []}}

    for _code, data, label in dataLoader:
        instanceBatch = model(data.to(device))
        results = classifier(instanceBatch)

        for _id, instance, result, _label in zip(_code, instanceBatch.detach().cpu().numpy(), results.detach().cpu().numpy(), label):
            if _label[1] == 1:
                instanceCNN["pos"]["code"].append(_id)
                instanceCNN["pos"]["result"].append(result)
                instanceCNN["pos"]["instance"].append(instance)
            else:
                instanceCNN["neg"]["code"].append(_id)
                instanceCNN["neg"]["result"].append(result)
                instanceCNN["neg"]["instance"].append(instance)

        currentNum += len(_code)

        print("%d/%d [%.2f%%]"%(currentNum, total, 100*currentNum/total), end="\r")
    print()
    pickle.dump(instanceCNN, open("instanceCNN.dump", "wb"))

    return None

def main():

    parser = argparse.ArgumentParser(description="Preprocess multi threading code.")
    parser.add_argument("--tsv", type=str, help="parameters recorded tsv text file path.")
    args = parser.parse_args()

    print("Train CNN model.")
    print("  Load variables")
    variables = dict(np.loadtxt(args.tsv, dtype=str, delimiter="\t"))
    sizeMicronXY = np.array(variables["sizeMicronXY"].split(','), dtype=int)
    intervalMicronXY = np.array(variables["intervalMicronXY"].split(','), dtype=int)
    resizePixelXY = np.array(variables["resizePixelXY"].split(','), dtype=int)
    loadInfosPath = variables["loadInfosPath"]
    numImage = int(variables["numImage"])
    numBatch = int(variables["numBatch"])
    numCycle = int(variables["numCycle"])
    learningRate = float(variables["learningRate"])
    modelPath = variables["modelPath"]
    del variables, args

    print("  Set WSI class.")
    cur = LoginWSIViewer()
    wsiluadinfo = GetWSIInfos(cur, "wsiluadinfo")

    patchImporterDic = {}
    for WSIID in wsiluadinfo.keys():
        patchImporterDic[WSIID] = WSIPatchImporter(wsiluadinfo[WSIID]["filepath"], wsiluadinfo[WSIID]["maskspec"], wsiluadinfo[WSIID]["maskanno"])
        patchImporterDic[WSIID].MakePatchCoordinates(sizeMicronXY, intervalMicronXY, resizePixelXY)

    print("  Load patch image dataset.")
    infoPack = pickle.load(open(loadInfosPath, "rb"))
    code = np.array(infoPack["code"])
    label = np.array(infoPack["label"])

    idxList = []
    for i in range(label.shape[1]):
        idxList.append(np.where(label.argmax(1) == i)[0])
    transformTensor = torchvision.transforms.ToTensor()

    print("  Set CNN model.")
    deviceName = "cuda:0"
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(deviceName if torch.cuda.is_available() else "cpu")
    if "cuda" in deviceName:
        os.environ['CUDA_LAUNCH_BLOCKING'] = deviceName.split(':')[-1]

    modelCNN = torchvision.models.efficientnet_v2_s(pretrained=True)
    classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),
                                     torch.nn.Linear(in_features=1280, out_features=2, bias=True),
                                     torch.nn.Softmax())
    modelCNN.classifier = classifier
    if os.path.isfile(modelPath):
        print("    Load CNN model parameters.")
        modelCNN.load_state_dict(torch.load(modelPath))
    modelCNN.train()
    modelCNN = modelCNN.to(device)

    criterionCNN = torch.nn.CrossEntropyLoss()
    optimizerCNN = torch.optim.Adam(modelCNN.parameters(), lr=learningRate)

    print("  Training start.")
    for _num in range(numCycle):

        idxRand = []
        for i in range(label.shape[1]):
            idxRand.append(np.random.choice(idxList[i], numImage))

        patchInfos = {"code" : code[np.hstack(idxRand)][:640], "label" : label[np.hstack(idxRand)][:640]}
        dataClassCycle = DatasetClassWSI(patchImporterDic, patchInfos, transformTensor)
        dataLoaderCycle = DataLoader(dataClassCycle, batch_size=numBatch, shuffle=True, drop_last=False, num_workers=4)

        TrainModel(modelCNN, dataLoaderCycle, criterionCNN, optimizerCNN, device, _num)
        torch.save(modelCNN.state_dict(), "modelCNN.pt")

    print("  Make instance.")
    modelCNN.eval()
    classifier = copy.deepcopy(modelCNN.classifier).to(device)
    modelCNN.classifier = torch.nn.Sequential()

    patchInfos = {"code" : code, "label" : label}
    dataClassAll = DatasetClassWSI(patchImporterDic, patchInfos, transformTensor)
    dataLoaderAll = DataLoader(dataClassAll, batch_size=numBatch, shuffle=False, drop_last=False, num_workers=4)

    MakeInstance(modelCNN, classifier, dataLoaderAll, device)
    print("  Instance data file is saved instanceCNN.dump.")

    return None

if __name__ == "__main__":
    main()
