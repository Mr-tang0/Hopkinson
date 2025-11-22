import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
from scipy.signal import savgol_filter, find_peaks, correlate
from numpy import argmax

# from main.Utilities.TwoDimVec import TwoDimVec


def zeroing(signals):
    """
            Not all experiments are born perfect, and some might be entirely offset.
            Thus, we can take the first value of each vector and move the entire
            vector upwards or downwards by that value
            (in an ideal experiment the first value will be 0).
    """
    try:
        for signal in signals:
            zeroer = signal[0]
            for i in range(len(signal)):
                signal[i] -= zeroer

    except Exception as e:
        print(e)


def manual_crop(time_ax, voltage, text, wave):
    """
    Manual graph cropping using GInput

    time_ax: the signals' time axis
    voltage: the signals' voltage
    text: The text the appears in the figure once opened
    wave: wave type: either "Incident", "Transmitted" or "Reflected".
    :return: time coordinate(s) of the user's cropping.

    Note:               "GInput" is a real troublemaker when used in a GUI (at least in this one),
                        which is why at failure the program will quit entirely.
    """

    fig, ax = plt.subplots()

    ax.plot(time_ax, voltage)
    ax.grid()
    plt.title(text)

    plt.xlabel("Time")
    plt.ylabel("Amplitude (V)")
    plt.legend(wave)

    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True, color='b', linewidth=1)

    try:
        # Get the selected coordinates from the graph:
        if wave == "Incident":
            cropping_coordinates = plt.ginput(n=3, show_clicks=True)

        else:
            cropping_coordinates = plt.ginput(n=1, show_clicks=True)

        plt.show()
        x, y = zip(*cropping_coordinates)
        # we are only interested in the X point values (Time) for the signal cutting:
        return x

    except Exception as e:
        #   If selection hasn't been made (interrupted), exit the program.
        #   Ginput is a dinosaur. It's bad for use and crushes instantly inside a GUI.
        print("User Interrupted",  e)
        exit()


# def auto_crop(update_logger, CA):
#     """
#         Automatic cropping of the signals through peak detection.
#
#         First we will create an array with the signals in absolute values.
#         Then, using SciPy's function "find_peaks()" we will find the peaks of the graph.
#         According to the first peak in the incident signal, we will crop the incident wave,
#         According to the second peak in the incident signal, we will crop the reflected wave.
#
#         This function manipulates existing signals and doesn't return anything.
#     """
#
#     #   For better peak-finding results, we will use smoothed signals, using the Savitzky - Golay filter:
#     sav_incid = savgol_filter(CA.incid.y, CA.smooth_value, 3)
#
#     #   A softer smoothing is applied on the transmitted signal since it is more sensitive than the incident signal:
#     sav_trans = savgol_filter(CA.trans.y, CA.smooth_value - 50, 3)
#
#     update_logger("\n...detecting signal peaks with Sav - Gol filter with "
#           + str(CA.smooth_value) + " window length (Curve Smoothing).")
#
#     #   Create absolute valued signals:
#     abs_incid = np.absolute(sav_incid)
#     abs_trans = np.absolute(sav_trans)
#
#     max_incid = max(sav_incid)
#
#     prominence = max_incid * CA.prominence_percent
#
#     #   Find the peaks in the Incident signal and take only the time stamps
#     peaks_incid, _ = find_peaks(abs_incid, prominence=prominence)
#
#     #   There should be at least 4 peaks in the signal,
#     #   if less than 4 have been found -> lower the prominence by one percent of its original value.
#     while len(peaks_incid) < 4:
#         prominence -= max_incid / 200
#         peaks_incid, _ = find_peaks(abs_incid, prominence=prominence)
#
#     #   For an estimated prominence, we will take half of the maximum value:
#     max_trans = max(sav_trans)
#     prominence = max_trans * CA.prominence_percent
#
#     #   Find the peaks in the Transmitted signal and take only the time stamps
#     peaks_trans, _ = find_peaks(abs_trans, prominence=prominence)
#
#     while len(peaks_trans) < 4:
#         prominence -= max_trans / 200
#         peaks_trans, _ = find_peaks(abs_trans, prominence=prominence)
#
#     update_logger("\n...Peaks detected in both signals. ")
#     #   The mode changes sign in some calculation, so a constant of 1 or -1 will be useful.
#     if CA.mode == "compression":
#         K = 1
#
#     else:
#         K = -1
#
#     #   The following will look for the point where the wave changes its sign from both sides,
#     #   Obtaining these points is crucial for determining where to crop the signal to export the different waves.
#
#     #   For incident wave:
#     incid_threshold = 0.05 * max_incid
#     incid_before_idx = peaks_incid[0]
#     while K * CA.incid_og.y[incid_before_idx] < - incid_threshold:
#         incid_before_idx -= 1
#
#     incid_after_idx = peaks_incid[0]
#     while K * CA.incid_og.y[incid_after_idx] < - incid_threshold:
#         incid_after_idx += 1
#
#     vcc_incid = CA.incid_og.y[incid_before_idx - CA.spacing: incid_after_idx + CA.spacing]
#     time_incid = CA.incid_og.x[incid_before_idx - CA.spacing: incid_after_idx + CA.spacing]
#
#     #   We want all three waves to be of the same vector size, so we will use the total time of the incident wave,
#     #   and only find where the other two waves begin.
#     signal_time = incid_after_idx - incid_before_idx
#
#     #   For reflected wave:
#     #reflected_before_idx = peaks_incid[1]
#     #while K * CA.incid_og.y[reflected_before_idx] > incid_threshold:
#     #    reflected_before_idx -= 1
#
#     # Calculate the beginning of reflected wave based on
#     # first strain gage position and soun velocity
#
#     reflected_before_idx = incid_before_idx + int(CA.first_gage*2/CA.sound_velocity*2e+6)
#
#     #   Total cropping time
#     reflected_after_idx = reflected_before_idx + signal_time
#     reflected_idx = reflected_before_idx
#     vcc_reflected = CA.incid_og.y[reflected_before_idx - CA.spacing: reflected_after_idx + CA.spacing]
#     time_reflected = CA.incid_og.x[reflected_before_idx - CA.spacing: reflected_after_idx + CA.spacing]
#
#     #   For transmitted wave:
#     trans_threshold = 0.01 * max_trans
#     trans_before_idx_real = peaks_trans[0]
#     while K * CA.trans_og.y[trans_before_idx_real] < - trans_threshold:
#         trans_before_idx_real -= 1
#
#     # Calculate the beginning of transition wave based on
#     # first strain gage position, second strain gage position
#     # and soun velocity, I add also length of the specimen
#     # better if it will be sound velocity of the sample, but we don't know it
#
#     trans_before_idx = incid_before_idx + int((CA.first_gage+CA.second_gage+CA.specimen_length)/CA.sound_velocity*2e+6)
#
#     # Here I add the CA.trans_shift parameter to understand how many points
#     # between transmitted signal risng and its beginning by sound velocity calculation
#     # It is important to cut the stress-strain curve
#
#     #   Total cropping time
#     trans_after_idx = trans_before_idx + signal_time
#     CA.trans_shift = trans_before_idx_real - trans_before_idx
#
#     '''
#             uncomment the following to display where the cropping occurs.
#
#     plt.clf()
#     plt.plot(og_time_incid, og_vcc_incid)
#     plt.plot(og_time_trans, og_vcc_trans)
#     plt.axvline(x=time_trans[incid_before_idx - spacing], linewidth=2, color='r')
#     plt.axvline(x=time_trans[before_idx - spacing], linewidth=2, color='k')
#     plt.axvline(x=time_trans[reflected_idx - spacing], linewidth=2, color='b')
#     plt.legend(["Incident", "Transmitted"])
#     plt.show()
#     '''
#
#     vcc_trans = CA.trans_og.y[trans_before_idx - CA.spacing: trans_after_idx + CA.spacing]
#     time_trans = CA.trans_og.x[trans_before_idx - CA.spacing: trans_after_idx + CA.spacing]
#
#     zeroing([time_incid, time_reflected, time_trans])
#
#     if CA.thermal_analysis:
#         og_volt_IR_EXP, og_time_IR_EXP = CA.IR_EXP.y, CA.IR_EXP.x
#
#         #   Use a copy of IR_EXP
#         time_IR_EXP = og_time_IR_EXP.copy()
#
#         #   The time difference that needs to be cut out from the IR signal.
#         dt = CA.first_gage / CA.sound_velocity + CA.incid_og.x[incid_before_idx] - CA.incid_og.x[CA.spacing]
#         cropping_idx = 0
#
#         for i in range(len(time_IR_EXP)):
#             if time_IR_EXP[i] >= dt:
#                 cropping_idx = i
#                 break
#
#         time_IR_EXP = time_IR_EXP[cropping_idx:]
#         volt_IR_EXP = og_volt_IR_EXP[cropping_idx:]
#
#         zeroing([time_IR_EXP])
#         incid = TwoDimVec(time_incid, vcc_incid)
#         trans = TwoDimVec(time_trans, vcc_trans)
#         refle = TwoDimVec(time_reflected, vcc_reflected)
#         IR_EXP = TwoDimVec(time_IR_EXP, volt_IR_EXP)
#         update_logger("\nAutomatic Cropping CMPLT.")
#         return incid, trans, refle, IR_EXP
#
#     incid = TwoDimVec(time_incid, vcc_incid)
#     trans = TwoDimVec(time_trans, vcc_trans)
#     refle = TwoDimVec(time_reflected, vcc_reflected)
#     IR_EXP = TwoDimVec()
#     update_logger("\nAutomatic Cropping CMPLT.")
#     cropping_points = [incid_before_idx, incid_after_idx,
#                        reflected_before_idx, reflected_after_idx,
#                        trans_before_idx, trans_after_idx]
#     return incid, trans, refle, IR_EXP, cropping_points
#
#
def mean_of_signal(update_logger, signal, prominence_percent, mode, spacing):
    """
        Calculate the mean value of the signal.
        (analytically this should be the value of the step).

    :param signal:  Some step signal of data.
    :return: the mean value of the peak.
    """

    #   Take absolute value of signal:
    abs_signal = np.absolute(signal)

    #   Smooth the signal to find peak:
    smooth_signal = savgol_filter(abs_signal, 51, 3)

    # find peak value of the signal:
    peak_value = max(abs_signal)

    if mode == "compression":
        k = 1
    else:
        k = -1

    #   Find the first peak of the signal, with a prominence of half the peak's value:
    peaks, _ = find_peaks(smooth_signal, prominence = peak_value * prominence_percent)
    if len(peaks) == 0:
        return -1
    peak = peaks[0]
    #   Start at peak and go backwards to find where the peak ends:
    idx = peak
    while idx > 0:
        if k * signal[idx] <= 0:
            break
        idx -= 1
    before_peak = idx

    #   Start at peak and forward to find where the peak ends:
    idx = peak
    while 0 < idx < len(signal):
        if k * signal[idx] <= 0:
            break
        idx += 1
    after_peak = idx

    # Crop the signal with some spacing
    cropped_signal = abs_signal[before_peak + spacing:after_peak - spacing]

    #   Take mean value of cropped signal:
    mean_value = np.mean(cropped_signal)

    #   Return the mean value:
    return mean_value


def cross_correlate_signals(update_logger, incident, transmitted, reflected, time_incid, time_trans, time_reflected, smooth_value):
    update_logger("\nSignal cross - corelation Initialized...")
    """
        This function uses Cross - Correlation to fix the given signals on each other.
        The "time_2" vector is the time vector that corresponds to the given "signal_2".

        The function only manipulates existing signals and returns nothing.

        signal_1 = incident
        signal_2 = transmitted
        signal_3 = reflected

        inputted signals should be dispersion corrected!
    """

    def autocorrelate(a, b):
        """
            SciPy's correlation is not starting at a time difference of 0, it starts at a negative time difference,
            closes to 0, and then goes positive. For this reason, we need to take the last half of the correlation
            result, and that should be the auto - correlation we are looking for.

        :param a: First signal to correlate to
        :param b: Second signal to be correlated to the first
        :return: fixed auto - correlated signal
        """
        result = correlate(a, b, mode='full')
        return result[result.size // 2:]

    abs_signal_1 = []
    abs_signal_2 = []
    abs_signal_3 = []

    time_1 = time_incid
    time_2 = time_trans
    time_3 = time_reflected

    signal_1 = savgol_filter(incident, smooth_value, 3)
    signal_2 = savgol_filter(transmitted, smooth_value, 3)
    signal_3 = savgol_filter(reflected, smooth_value, 3)

    for value in signal_1:
        abs_signal_1.append(abs(value))

    for value in signal_2:
        abs_signal_2.append(abs(value))

    for value in signal_3:
        abs_signal_3.append(abs(value))

    '''
    #   Auto - Correlated Signal (acs) of the Incident signal and the Transmitted signal:
    acs = autocorrelate(abs_signal_1, abs_signal_2)

    #   Get the position of the maximum value in the Auto - Correlated Signal:
    correlated_position = np.argmax(acs)
    correlated_time = time_2[correlated_position]

    if correlated_position != 0:
        #   Align signal_2 to signal_1 according to the ACS result:
        time_1 = time_1[:-correlated_position]
        time_2 = time_2[correlated_position:]

        signal_1 = CA.fixed_incident[:-correlated_position]
        signal_2 = CA.fixed_transmitted[correlated_position:]

    for i in range(len(time_2)):
        time_2[i] -= correlated_time

    '''

    #   Auto - Correlate signal (acs) of the Incident signal and the Reflected signal:
    acs = autocorrelate(abs_signal_3, abs_signal_1)

    #   Get the position of the maximum value in the Auto - Correlated Signal:
    correlated_position = argmax(acs)
    correlated_time = time_3[correlated_position]

    if correlated_position != 0:
        #   Align signal_2 to signal_1 according to the ACS result:
        time_1 = time_1[:-correlated_position]
        time_2 = time_2[:-correlated_position]
        time_3 = time_3[correlated_position:]

        signal_1 = incident[:-correlated_position]
        signal_2 = transmitted[:-correlated_position]
        signal_3 = reflected[correlated_position:]

        for i in range(len(time_3)):
            time_2[i] -= correlated_time / 2
            time_3[i] -= correlated_time

        incident = signal_1
        transmitted = signal_2
        reflected = signal_3

    time_incid = time_1
    time_trans = time_2
    time_reflected = time_3
    update_logger("\nCross - Correlation CMPLT.")
    return incident, transmitted, reflected, time_incid, time_trans, time_reflected


def crop_signal(CA, mode, before_crop, after_crop, reflected_crop=0.0, transmitted_crop=0.0):
        """
                This function crops the signal, first crop is from the left and second is from the right.
                In the second crop the function will return the time difference between
                the crops for further automatic cropping.

                CA = CoreAnalyzer
        """
        if mode == "before":
            for i in range(len(CA.incid.x)):
                if CA.incid.x[i] >= before_crop:

                    CA.incid.crop_from_index(i)
                    CA.trans.crop_from_index(i)

                    for j in range(len(CA.incid.x)):
                        CA.incid.x[j] -= before_crop
                        CA.trans.x[j] -= before_crop

                    break

        elif mode == "after":
            for i in range(len(CA.incid.x)):
                if CA.incid.x[i] >= after_crop:
                    CA.incid.x = CA.incid.x[:i]
                    CA.incid.y = CA.incid.y[:i]

                    break
            align = CA.incid.x[0]
            for j in range(len(CA.incid.x)):
                CA.incid.x[j] -= align

        elif mode == "reflected":
            time_period = after_crop

            for i in range(len(CA.incid_og.x)):
                if CA.incid_og.x[i] >= reflected_crop:
                    reflected_crop_idx = i
                    break

            tpp = CA.incid_og.x[1] - CA.incid_og.x[0]
            time_period = round(time_period / tpp)
            CA.refle.x = CA.incid_og.x[reflected_crop_idx:reflected_crop_idx + time_period].copy()
            CA.refle.y = CA.incid_og.y[reflected_crop_idx:reflected_crop_idx + time_period].copy()

            align = CA.refle.x[0]
            for j in range(len(CA.refle.x)):
                CA.refle.x[j] -= align

        elif mode == "transmitted":
            time_period = after_crop

            for i in range(len(CA.trans.x)):
                if CA.trans.x[i] >= transmitted_crop:
                    transmitted_crop_idx = i
                    break

            tpp = CA.incid_og.x[1] - CA.incid_og.x[0]
            time_period = round(time_period / tpp)

            CA.trans.x = CA.trans.x[transmitted_crop_idx:transmitted_crop_idx + time_period]
            CA.trans.y = CA.trans.y[transmitted_crop_idx:transmitted_crop_idx + time_period]

            align = CA.trans.x[0]
            for j in range(len(CA.trans.x)):
                CA.trans.x[j] -= align