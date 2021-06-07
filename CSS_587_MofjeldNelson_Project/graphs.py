# Carl Mofjeld, Drew Nelson
# CSS 587 - Advance Topics in Computer Vision
# Spring 2021
#
# Some quick code to help generate plots based on the results
# created from the main application

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Read data from a csv file and extract the error and timing results from each
# method (four point, six point, seven point, eight point )
def GetDataFromTrial(filePath):
    fourPointResults = { "errors": [ ], "times": [ ] }
    sixPointResults = { "errors": [ ], "times": [ ] }
    sevenPointResults = { "errors": [ ], "times": [ ] }
    eightPointResults = { "errors": [ ], "times": [ ] }

    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            iteration = row[0]
            method = row[1]
            error = math.log10(float(row[2])) # Apply Log10 here since authors are doing the same
            time = int(row[3])
            
            results = { }
            if method == "FourPoint":
                results = fourPointResults
            elif method == "SixPoint":
                results = sixPointResults
            elif method == "CV_SevenPoint":
                results = sevenPointResults
            elif method == "CV_EightPoint":
                results = eightPointResults

            if len(results) > 0:
                results["errors"].append(error)
                results["times"].append(time)

    rawErrorResults = [ ]

    averageFourPointTime = 0
    averageSixPointTime = 0
    averageSevenPointTime = 0
    averageEightPointTime = 0

    numOfIterations = len(fourPointResults["errors"])

    for i in range(numOfIterations):
        fourPointError = fourPointResults["errors"][i]
        sixPointError = sixPointResults["errors"][i]
        sevenPointError = sevenPointResults["errors"][i]
        eightPointError = eightPointResults["errors"][i]

        fourPointTime = fourPointResults["times"][i]
        sixPointTime = sixPointResults["times"][i]
        sevenPointTime = sevenPointResults["times"][i]
        eightPointTime = eightPointResults["times"][i]

        iteration = [ fourPointError, sixPointError, sevenPointError, eightPointError ]
        rawErrorResults.append(iteration)

        averageFourPointTime = averageFourPointTime + fourPointTime
        averageSixPointTime = averageSixPointTime + sixPointTime
        averageSevenPointTime = averageSevenPointTime + sevenPointTime
        averageEightPointTime = averageEightPointTime + eightPointTime

    averageFourPointTime = averageFourPointTime / numOfIterations
    averageSixPointTime = averageSixPointTime / numOfIterations
    averageSevenPointTime = averageSevenPointTime / numOfIterations
    averageEightPointTime = averageEightPointTime / numOfIterations

    errorResults = np.array(rawErrorResults)

    print("Average Four Point time: " + str(averageFourPointTime) + " microseconds")
    print("Average Six Point time: " + str(averageSixPointTime) + " microseconds")
    print("Average Seven Point time: " + str(averageSevenPointTime) + " microseconds")
    print("Average Eight Point time: " + str(averageEightPointTime) + " microseconds")

    return {
        "rawResults": [ fourPointResults, sixPointResults, sevenPointResults, eightPointResults ],
        "formattedErrorResults": errorResults,
        "timeAverages": [ averageFourPointTime, averageSixPointTime, averageSevenPointTime, averageEightPointTime ]
    }

# Attempt to plot the results from the zero-noise trials. Plots as a probability density graph
def plotZeroNoiseTrial():
    zeroNoiseTrailDetails = GetDataFromTrial('final_results/zero_noise_trail.csv')
    zeroNoiseDf = pd.DataFrame(zeroNoiseTrailDetails["formattedErrorResults"], columns = ['Four Point','Six Point','Seven Point', 'Eight Point'])
    densityPlot = zeroNoiseDf.plot.kde()
    densityPlot.set_xlabel("Log10 Frobenius Norm Error")
    densityPlot.set_ylabel("Probability")
    plt.title("Zero Noise Trial")
    plt.savefig("zero-noise.png")
    plt.show()
    
# format/combine the data from all of the noisy trials. Essentially, try to combine all of the
# results into one table
def formatNoisy(trialResults):
    allFourPoints = [ ]
    allSixPoints = [ ]
    allSevenPoints = [ ]
    allEightPoints = [ ]
    for trialResult in trialResults:
        filePath = trialResult["filePath"] 
        results = GetDataFromTrial(filePath)
        fourPointResults = results["rawResults"][0]["errors"]
        sixPointResults = results["rawResults"][1]["errors"]
        sevenPointResults = results["rawResults"][2]["errors"]
        eightPointResults = results["rawResults"][3]["errors"]

        allFourPoints.append(fourPointResults)
        allSixPoints.append(sixPointResults)
        allSevenPoints.append(sevenPointResults)
        allEightPoints.append(eightPointResults)

    return [ allFourPoints, allSixPoints, allSevenPoints, allEightPoints ]

# Attempt to plot the results from all of the noisy trials. Plots as grouped boxplot char
def plotNoiseTrials():
    trialResults = [
        { "filePath": "final_results/noise_trial_0.csv", "name": 0 },
        { "filePath": "final_results/noise_trial_0_01.csv", "name": 0.01 },
        { "filePath": "final_results/noise_trial_0_1.csv", "name": 0.1 },
        { "filePath": "final_results/noise_trial_0_5.csv", "name": 0.5 },
        { "filePath": "final_results/noise_trial_1.csv", "name": 1 },
        { "filePath": "final_results/noise_trial_2.csv", "name": 2 }
    ]

    details = formatNoisy(trialResults)

    fourPointColor ='#FF0000'
    sixPointColor ='#00FF00'
    sevenPointColor ='#0000FF'
    eightPointColor = '#FF00FF'

    fourPointData = details[0]
    sixPointData = details[1]
    sevenPointData = details[2]
    eightPointData = details[3]

    ticks = ['0', '0.01', '0.1', '0.5', '1', '2']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    bpFourPoint = plt.boxplot(fourPointData, positions=np.array(range(len(fourPointData)))*2.0-0.4, sym='', widths=0.1)
    bpSixPoint = plt.boxplot(sixPointData, positions=np.array(range(len(sixPointData)))*2.0-0.2, sym='', widths=0.1)
    bpSevenPoint = plt.boxplot(sevenPointData, positions=np.array(range(len(sevenPointData)))*2.0+0.2, sym='', widths=0.1)
    bpEightPoint = plt.boxplot(eightPointData, positions=np.array(range(len(eightPointData)))*2.0+0.4, sym='', widths=0.1)
    
    set_box_color(bpFourPoint, fourPointColor)
    set_box_color(bpSixPoint, sixPointColor)
    set_box_color(bpSevenPoint, sevenPointColor)
    set_box_color(bpEightPoint, eightPointColor)

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=fourPointColor, label='Four Point Method')
    plt.plot([], c=sixPointColor, label='Six Point Method')
    plt.plot([], c=sevenPointColor, label='Seven Point Method')
    plt.plot([], c=eightPointColor, label='Eight Point Method')
    plt.legend()

    plt.xlabel("Noise Level (pixels)")
    plt.ylabel("Log10 Frobenius Norm Error")

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.title('Noisy Trials')
    plt.savefig("noisy.png")
    plt.show()

def main(): 
    plotZeroNoiseTrial()
    plotNoiseTrials()

if __name__ == "__main__":
    main()