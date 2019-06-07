from __future__ import division
import math
import numpy as np
USE_PIL = True
USE_CV2 = False
USE_SCIPY = False
try:
    import PIL
    from PIL import Image
    raise ImportError
except ImportError:
    USE_PIL = False
if not USE_PIL:
    USE_CV2 = True
    try:
        import cv2
    except ImportError:
        USE_CV2 = False
if not USE_PIL and not USE_CV2:
    USE_SCIPY = True
    try:
        import scipy
        from scipy import misc
    except ImportError:
        USE_SCIPY = False
        raise RuntimeError("Erro")


class ImageReadWrite(object):

    def read(self, filename):
        if USE_PIL:
            color_im = PIL.Image.open(filename)
            grey = color_im.convert('L')
            return np.array(grey, dtype=np.uint8)
        elif USE_CV2:
            return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        elif USE_SCIPY:
            greyscale = True
            float_im = scipy.misc.imread(filename, greyscale)
            im = np.array(float_im, dtype=np.uint8)
            return im

    def write(self, filename, array):
        if USE_PIL:
            im = PIL.Image.fromarray(array)
            im.save(filename)
        elif USE_SCIPY:
            scipy.misc.imsave(filename, array)
        elif USE_CV2:
            cv2.imwrite(filename, array)


class _OtsuPyramid(object):

    def load_image(self, im, bins=256):
        if not type(im) == np.ndarray:
            raise ValueError(
                'must be passed numpy array. Got ' + str(type(im)) +
                ' instead'
            )
        if im.ndim == 3:
            raise ValueError(
                'image must be greyscale (and single value per pixel)'
            )
        self.im = im
        hist, ranges = np.histogram(im, bins)
        hist = [int(h) for h in hist]
        histPyr, omegaPyr, muPyr, ratioPyr = \
            self._create_histogram_and_stats_pyramids(hist)
        self.omegaPyramid = [omegas for omegas in reversed(omegaPyr)]
        self.muPyramid = [mus for mus in reversed(muPyr)]
        self.ratioPyramid = ratioPyr
        
    def _create_histogram_and_stats_pyramids(self, hist):

        bins = len(hist)
        ratio = 2
        reductions = int(math.log(bins, ratio))
        compressionFactor = []
        histPyramid = []
        omegaPyramid = []
        muPyramid = []
        for _ in range(reductions):
            histPyramid.append(hist)
            reducedHist = [sum(hist[i:i+ratio]) for i in range(0, bins, ratio)]

            hist = reducedHist

            bins = bins // ratio
            compressionFactor.append(ratio)

        compressionFactor[0] = 1
        for hist in histPyramid:
            omegas, mus, muT = \
                self._calculate_omegas_and_mus_from_histogram(hist)
            omegaPyramid.append(omegas)
            muPyramid.append(mus)
        return histPyramid, omegaPyramid, muPyramid, compressionFactor

    def _calculate_omegas_and_mus_from_histogram(self, hist):

        probabilityLevels, meanLevels = \
            self._calculate_histogram_pixel_stats(hist)
        bins = len(probabilityLevels)

        ptotal = float(0)
        omegas = []
        for i in range(bins):
            ptotal += probabilityLevels[i]
            omegas.append(ptotal)
        mtotal = float(0)
        mus = []
        for i in range(bins):
            mtotal += meanLevels[i]
            mus.append(mtotal)
        muT = float(mtotal)
        return omegas, mus, muT

    def _calculate_histogram_pixel_stats(self, hist):

        bins = len(hist)
        N = float(sum(hist))

        hist_probability = [hist[i] / N for i in range(bins)]
        pixel_mean = [i * hist_probability[i] for i in range(bins)]
        return hist_probability, pixel_mean


class OtsuFastMultithreshold(_OtsuPyramid):


    def calculate_k_thresholds(self, k):
        self.threshPyramid = []
        start = self._get_smallest_fitting_pyramid(k)
        self.bins = len(self.omegaPyramid[start])
        thresholds = self._get_first_guess_thresholds(k)
        deviate = self.bins // 2
        for i in range(start, len(self.omegaPyramid)):
            omegas = self.omegaPyramid[i]
            mus = self.muPyramid[i]
            hunter = _ThresholdHunter(omegas, mus, deviate)
            thresholds = \
                hunter.find_best_thresholds_around_estimates(thresholds)
            self.threshPyramid.append(thresholds)
            scaling = self.ratioPyramid[i]
            deviate = scaling
            thresholds = [t * scaling for t in thresholds]
        return [t // scaling for t in thresholds]

    def _get_smallest_fitting_pyramid(self, k):

        for i, pyramid in enumerate(self.omegaPyramid):
            if len(pyramid) >= k:
                return i

    def _get_first_guess_thresholds(self, k):

        kHalf = k // 2
        midway = self.bins // 2
        firstGuesses = [midway - i for i in range(kHalf, 0, -1)] + [midway] + \
            [midway + i for i in range(1, kHalf)]
        firstGuesses.append(self.bins - 1)
        return firstGuesses[:k]

    def apply_thresholds_to_image(self, thresholds, im=None):
        if im is None:
            im = self.im
        k = len(thresholds)
        bookendedThresholds = [None] + thresholds + [None]
        greyValues = [0] + [int(256 / k * (i + 1)) for i in range(0, k - 1)] \
            + [255]
        greyValues = np.array(greyValues, dtype=np.uint8)
        finalImage = np.zeros(im.shape, dtype=np.uint8)
        for i in range(k + 1):
            kSmall = bookendedThresholds[i]
            bw = np.ones(im.shape, dtype=np.bool8)
            if kSmall:
                bw = (im >= kSmall)
            kLarge = bookendedThresholds[i + 1]
            if kLarge:
                bw &= (im < kLarge)
            greyLevel = greyValues[i]
            greyImage = bw * greyLevel
            finalImage += greyImage
        return finalImage


class _ThresholdHunter(object):

    def __init__(self, omegas, mus, deviate=2):
        self.sigmaB = _BetweenClassVariance(omegas, mus)
        self.bins = self.sigmaB.bins
        self.deviate = deviate

    def find_best_thresholds_around_estimates(self, estimatedThresholds):
        bestResults = (
            0, estimatedThresholds, [0 for t in estimatedThresholds]
        )
        bestThresholds = estimatedThresholds
        bestVariance = 0
        for thresholds in self._jitter_thresholds_generator(
                estimatedThresholds, 0, self.bins):
            variance = self.sigmaB.get_total_variance(thresholds)
            if variance == bestVariance:
                if sum(thresholds) < sum(bestThresholds):
                    bestThresholds = thresholds
            elif variance > bestVariance:
                bestVariance = variance
                bestThresholds = thresholds
        return bestThresholds

    def find_best_thresholds_around_estimates_experimental(self, estimatedThresholds):
        estimatedThresholds = [int(k) for k in estimatedThresholds]
        if sum(estimatedThresholds) < 10:
            return self.find_best_thresholds_around_estimates_old(
                estimatedThresholds
            )
        print('estimated', estimatedThresholds)
        fxn_to_minimize = lambda x: -1 * self.sigmaB.get_total_variance(
            [int(k) for k in x]
        )
        bestThresholds = scipy.optimize.fmin(
            fxn_to_minimize, estimatedThresholds
        )
        bestThresholds = [int(k) for k in bestThresholds]
        print('bestTresholds', bestThresholds)
        return bestThresholds

    def _jitter_thresholds_generator(self, thresholds, min_, max_):
        pastThresh = thresholds[0]
        if len(thresholds) == 1:
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                if thresh < min_ or thresh >= max_:
                    continue
                yield [thresh]
        else:
            thresholds = thresholds[1:]
            m = len(thresholds)
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                if thresh < min_ or thresh + m >= max_:
                    continue
                recursiveGenerator = self._jitter_thresholds_generator(
                    thresholds, thresh + 1, max_
                )
                for otherThresholds in recursiveGenerator:
                    yield [thresh] + otherThresholds


class _BetweenClassVariance(object):

    def __init__(self, omegas, mus):
        self.omegas = omegas
        self.mus = mus
        self.bins = len(mus)
        self.muTotal = sum(mus)

    def get_total_variance(self, thresholds):
        thresholds = [0] + thresholds + [self.bins - 1]
        numClasses = len(thresholds) - 1
        sigma = 0
        for i in range(numClasses):
            k1 = thresholds[i]
            k2 = thresholds[i+1]
            sigma += self._between_thresholds_variance(k1, k2)
        return sigma

    def _between_thresholds_variance(self, k1, k2):
        omega = self.omegas[k2] - self.omegas[k1]
        mu = self.mus[k2] - self.mus[k1]
        muT = self.muTotal
        return omega * ((mu - muT)**2)


if __name__ == '__main__':
    filename = 'muda.jpg'
    dot = filename.index('.')
    prefix, extension = filename[:dot], filename[dot:]
    imager = ImageReadWrite()
    im = imager.read(filename)
    otsu = OtsuFastMultithreshold()
    otsu.load_image(im)
    for k in [1, 2, 3, 4, 5, 6]:
        savename = prefix + '_crushed_' + str(k) + extension
        kThresholds = otsu.calculate_k_thresholds(k)
        print(kThresholds)
        crushed = otsu.apply_thresholds_to_image(kThresholds)
        imager.write(savename, crushed)
