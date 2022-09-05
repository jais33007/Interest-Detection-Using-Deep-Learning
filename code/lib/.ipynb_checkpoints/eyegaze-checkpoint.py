import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import matplotlib.cm as cm

import json
import seaborn as sns
from progressbar import ProgressBar
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


def isMinimumFixation(X, Y, mfx):
    if max([max(X) - min(X), max(Y) - min(Y)]) < mfx:
        return True
    return False


def isFixation(X, Y, mfx):
    if max([max(X) - min(X), max(Y) - min(Y)]) < mfx:
        return True
    return False


def detectFixations(
        times, X, Y,
        min_concat_gaze_count=9,
        min_fixation_size=50,
        max_fixation_size=80,
        save_path=""):

    fixations = []
    i = 0
    j = 0
    while max([i, j]) < len(times) - min_concat_gaze_count:
        X_ = list(X[i:i+min_concat_gaze_count])
        Y_ = list(Y[i:i+min_concat_gaze_count])
        if isMinimumFixation(X_, Y_, min_fixation_size):
            j = i + min_concat_gaze_count - 1
            c = 0
            begin_time = times[i]
            end_time = times[j]
            while(c < min_concat_gaze_count and j < len(times)):
                X_.append(X[j])
                Y_.append(Y[j])
                if max([max(X_) - min(X_), max(Y_) - min(Y_)]) > max_fixation_size:
                    # X[j]Y[j] is out of max_fixation_size
                    if c == 0:
                        # X[j]Y[j] will be next minimum fixation
                        i = j
                    X_.pop()
                    Y_.pop()
                    c += 1
                else:
                    c = 0
                    end_time = times[j]
                j += 1
            fixations.append([begin_time, np.mean(X_), np.mean(Y_), end_time - begin_time])
        i += 1

    if len(fixations) < 2:
        return np.array([])

    # saccade detection
    lengths = []
    angles = []
    durations = []
    for i in range(1, len(fixations)):
        delta_x = fixations[i][1] - fixations[i-1][1]
        delta_y = fixations[i][2] - fixations[i-1][2]
        delta_t = fixations[i][0] - (fixations[i-1][0] + fixations[i-1][3])
        lengths.append(math.sqrt(delta_x*delta_x + delta_y*delta_y))
        angles.append(math.atan2(delta_y, delta_x))
        durations.append(delta_t)
    saccades = np.vstack((lengths, angles, np.array(lengths)/np.array(durations))).T
    results = np.hstack((fixations[1:], saccades))
    if save_path != "":
        np.savetxt(save_path, results, delimiter=',', header=(
                   '#timestamp,fixation_x,fixation_y,fixation_duration,' +
                   'saccade_length,saccade_angle,saccade_velocity'))
    return results


def plotScanPath(
        X, Y, durations, figsize=(30, 15),
        bg_image="", save_path="", halfPage=False):
    plt.figure(figsize=figsize)
    if bg_image != "":
        img = mpimg.imread(bg_image)
        plt.imshow(img)
        if halfPage:
            plt.xlim(150, 1000)
        else:
            plt.xlim(0, len(img[0]))
        plt.ylim(len(img), 0)
    scale = float(figsize[0]) / 40.0

    plt.plot(X, Y, "-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
    plt.scatter(X, Y, durations*scale, c="b", zorder=2, alpha=0.3)
    plt.scatter(X[0], Y[0], durations[0]*scale, c="g", zorder=2, alpha=0.6)
    plt.scatter(X[-1], Y[-1], durations[-1]*scale, c="r", zorder=2, alpha=0.6)

    if save_path != "":
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()


def plotHeatmap(
        X, Y, durations, figsize=(30, 15),
        bg_image="", save_path="", data_save_path=""):
    values = calcHeatmap(
            X, Y, durations, figsize, bg_image, save_path, data_save_path)
    plotHeatmapFromExported(
            values, figsize, bg_image, save_path, data_save_path)


def calcHeatmap(
        X, Y, durations, figsize=(30, 15),
        bg_image="", save_path="", data_save_path=""):
    if bg_image != "":
        img = mpimg.imread(bg_image)

    gx, gy = np.meshgrid(np.arange(0, len(img[0])), np.arange(0, len(img)))
    values = np.zeros((len(img), len(img[0])))
    for i in range(len(X)):
        values += mlab.bivariate_normal(gx, gy, 50, 50, X[i], Y[i])*durations[i]/2.0
    return values/np.max(values)


def plotHeatmapFromExported(
        values, figsize=(30, 15), bg_image="",
        save_path="", data_save_path=""):
    plt.figure(figsize=figsize)
    if bg_image != "":
        img = mpimg.imread(bg_image)
        plt.imshow(img)
        plt.xlim(0, len(img[0]))
        plt.ylim(len(img), 0)

    masked = np.ma.masked_where(values < 0.05, values)
    cmap = cm.jet
    cmap.set_bad('white', 1.)
    plt.imshow(masked, alpha=0.4, cmap=cmap)

    if save_path != "":
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    if data_save_path != "":
        np.savetxt(data_save_path, values, delimiter=",", fmt="%f")


        
def plot(
        x, y, duration, figsize=(30, 15),
        image="", save_path="", halfPage=False):
    plt.figure(figsize=figsize)
    if image != "":
        img = mpimg.imread(image)
        plt.imshow(img)
        if halfPage:
            plt.xlim(150, 1000)
        else:
            plt.xlim(0, len(img[0]))
        plt.ylim(len(img), 0)
    scale = float(figsize[0]) / 40.0
    colors = [hsv2rgb((float(i)/len(x))*180, 0.8, 0.8, asCode=True) for i in range(len(x))]

#    plt.plot(x, y,"-", c="blue", linewidth=scale, zorder=1, alpha=0.8)
#    plt.scatter(x, y, duration*scale, c="b", zorder=2, alpha=0.3)
#    plt.scatter(x[0], y[0], duration[0]*scale, c="g", zorder=2, alpha=0.6)
#    plt.scatter(x[-1], y[-1], duration[-1]*scale, c="r", zorder=2, alpha=0.6)

    # color
    plt.plot(x, y, "-", c="gray", linewidth=scale, zorder=1, alpha=0.8)
    plt.scatter(x, y, duration*scale, c=colors, zorder=2, alpha=0.5)

    if save_path != "":
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()
        
        
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hsv2rgb(h, s, v, asCode=False):
    while h < 0:
        h += 360
    h %= 360
    if s == 0:
        v = int(round(v*255))
        return [v, v, v]

    hi = int(h / 60) % 6
    f = h / 60 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if hi == 0:
        rgb = [v, t, p]
    elif hi == 1:
        rgb = [q, v, p]
    elif hi == 2:
        rgb = [p, v, t]
    elif hi == 3:
        rgb = [p, q, v]
    elif hi == 4:
        rgb = [t, p, v]
    elif hi == 5:
        rgb = [v, p, q]

    res = [int(round(x*255)) for x in rgb]
    if asCode:
        return "#"+hex(res[0])[2:]+hex(res[1])[2:]+hex(res[2])[2:]
    else:
        return res

