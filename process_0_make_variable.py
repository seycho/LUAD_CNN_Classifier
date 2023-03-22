import argparse
import numpy as np

def main():

    parser = argparse.ArgumentParser(description="Prepare variables text file.")
    parser.add_argument("-n", type=str, help="Cycle number")
    args = parser.parse_args()

    num = int(args.n)

    print("Make variables text file.")
    var1 = {}
    var1["sizeMicronXY"] = "200,200"
    var1["intervalMicronXY"] =  "100,100"
    var1["resizePixelXY"] = "128,128"
    var1["numThread"] = "16"
    var1["saveInfosPath"] = "infoPack_%d.dump"%num
    var1["loadInfosPath"] = "infoPack_0.dump"
    var1["refLabelPath"] = "resultsMIL_%d.dump"%(num-1)
    np.savetxt("variable_1_prepare_dataset.tsv", np.array(list(var1.items())), fmt="%s", delimiter="\t")
    print("  Process 1 variables text file is saved variable_1_prepare_dataset.tsv")

    var2 = {}
    var2["sizeMicronXY"] = "200,200"
    var2["intervalMicronXY"] =  "100,100"
    var2["resizePixelXY"] = "128,128"
    var2["loadInfosPath"] = var1["saveInfosPath"]
    var2["numImage"] = "32000"
    var2["numBatch"] = "32"
    var2["numCycle"] = "1"
    var2["learningRate"] = "0.0001"
    var2["modelPath"] = "modelCNN.pt"
    np.savetxt("variable_2_CNN_training.tsv", np.array(list(var2.items())), fmt="%s", delimiter="\t")
    print("  Process 2 variables text file is saved variable_2_CNN_training.tsv")

    var3 = {}
    var3["saveLabelPath"] = "resultsMIL_%d.dump"%num
    var3["numCycle"] = "10"
    var3["learningRate"] = "0.001"
    var3["modelPath"] = "modelMIL.pt"
    np.savetxt("variable_3_MIL_training.tsv", np.array(list(var3.items())), fmt="%s", delimiter="\t")
    print("  Process 3 variables text file is saved variable_3_MIL_training.tsv")

    var4 = {}
    var4["sizeMicronXY"] = "200,200"
    var4["intervalMicronXY"] =  "100,100"
    var4["resizePixelXY"] = "128,128"
    var4["saveAnalyPath"] = "analysis/results_%s/"%num
    var4["loadInfosPath"] = "infoPack_0.dump"
    var4["loadLabelPath"] = var3["saveLabelPath"]
    np.savetxt("variable_4_analysis_results.tsv", np.array(list(var4.items())), fmt="%s", delimiter="\t")
    print("  Process 4 variables text file is saved variable_4_analysis_results.tsv")
    print()

    return None

if __name__ == "__main__":
    main()
