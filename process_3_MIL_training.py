import numpy as np
import argparse, torch, pickle, os

# dsmil code (https://github.com/binli123/dsmil-wsi)
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 1) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 

def TrainModel(model, instance, criterion, optimizer, num):

    for _type in instance.keys():
        resultTensor = torch.Tensor(np.array(instance[_type]["result"]))
        instanceTensor = torch.Tensor(np.array(instance[_type]["instance"]))

        results, label, _ = model(instanceTensor, resultTensor)

        if _type == "pos":
            target = torch.Tensor([[0, 1]])
        else:
            target = torch.Tensor([[1, 0]])

        optimizer.zero_grad()
        loss = criterion(results, target)

        loss.backward()
        optimizer.step()

    return None

def MakeResult(model, instance):

    correctMILDic = {}
    for _type in instance.keys():
        code = instance[_type]["code"]
        resultArray = np.array(instance[_type]["result"])
        instanceArray = np.array(instance[_type]["instance"])
        resultTensor = torch.Tensor(resultArray)
        instanceTensor = torch.Tensor(instanceArray)

        results, label, _ = model(instanceTensor, resultTensor)

        correctMILDic.update(dict(zip(code, label.detach().numpy())))

    return correctMILDic

def main():

    parser = argparse.ArgumentParser(description="Preprocess multi threading code.")
    parser.add_argument("--tsv", type=str, help="parameters recorded tsv text file path.")
    args = parser.parse_args()

    print("Train MIL model.")
    print("  Load variables")
    variables = dict(np.loadtxt(args.tsv, dtype=str, delimiter="\t"))
    saveLabelPath = variables["saveLabelPath"]
    numCycle = int(variables["numCycle"])
    learningRate = float(variables["learningRate"])
    modelPath = variables["modelPath"]

    print("  Load instance data.")
    instanceCNN = pickle.load(open("instanceCNN.dump", "rb"))

    print("  Set MIL model.")
    modelMIL = BClassifier(1280, 2)
    if os.path.isfile(modelPath):
        print("    Load MIL model parameters.")
        modelMIL.load_state_dict(torch.load(modelPath))

    criterionMIL = torch.nn.CrossEntropyLoss()
    optimizerMIL = torch.optim.Adam(modelMIL.parameters(), lr=learningRate)

    print("  Training start.")
    for _idx in range(numCycle):
        TrainModel(modelMIL, instanceCNN, criterionMIL, optimizerMIL, _idx)
        print("%d / %d"%(_idx, numCycle), end="\r")
        torch.save(modelMIL.state_dict(), "modelMIL.pt")

    correctMILDic = MakeResult(modelMIL, instanceCNN)
    pickle.dump(correctMILDic, open(saveLabelPath, "wb"))
    print("  Results file is saved %s."%saveLabelPath)
    print()

    return None

if __name__ == "__main__":
    main()