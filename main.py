import os
import re
import numpy as np

from RFP import rfp_poles

# Parameters (better add way for user to set them)
responses_dir = "RESPONSES"
parameters_dir = os.path.curdir
reference_channel = 2
sampling_freq = 100000
sampling_time = 0.1

df = 1 / sampling_time
response_length = int(sampling_time * sampling_freq)

# Loading responses
file_names = os.listdir(responses_dir)
strings, channels = [], []
for name in file_names:
    match = re.match(r"^(\d+)_(\d+).txt$", name)
    if match == None:
        continue
    string, channel = map(int, match.groups())
    if string not in strings:
        strings.append(string)
    if channel not in channels:
        channels.append(channel)
strings.sort()
channels.sort()
reference_channel_index = channels.index(reference_channel)

responses = [[[]] * len(channels) for _ in strings]
for string in range(len(strings)):
    for channel in range(len(channels)):
        file_path = os.path.join(
            responses_dir, f"{strings[string]}_{channels[channel]}.txt"
        )
        response = [float(line) for line in open(file_path)]
        if len(response) > response_length:
            response = response[:response_length]
        else:
            response = response + [0] * (response_length - len(response))
        responses[string][channel] = response
responses = np.array(responses, dtype=np.cdouble)
references = responses[:, reference_channel_index]
responses = np.delete(responses, reference_channel_index, 1)

# Apply windows???

Hs = (
    np.arange(0, df * references.shape[-1], df)
    * np.fft.fft(responses)
    / np.max(np.fft.fft(references)[:, np.newaxis], 0)
)
borders = [
    30,
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    4000,
    5000,
]  # arbitrary frequency borders. needed because rfp can find maximum of 15 modes in an interval
poles = None
for i in range(len(borders) - 1):
    new_poles = []
    omega = 2 * np.pi * np.arange(borders[i], borders[i + 1], df)
    for n in range(30, -2, -2):
        if n == 0:
            new_poles = []
            break
        new_poles = rfp_poles(
            omega,
            Hs[:, :, int(borders[i] // df) : int(borders[i + 1] // df)],
            30,
            n,
        )
        if len(new_poles) * 2 == n:
            break
    if poles is None:
        poles = new_poles
    else:
        poles = np.append(poles, new_poles)

poles = poles[  # pyright: ignore[reportOptionalSubscript]
    np.argsort(
        np.imag(  # pyright: ignore[reportCallIssue]
            poles  # pyright: ignore[reportArgumentType]
        )
    )
]

pole_freqs = np.imag(poles) / (2 * np.pi)
channel_amplitudes = np.zeros((0, 6))
strings_amplitudes = np.zeros((0, 64))
masses = np.array([])


for pole in poles:
    pole_index = int(np.imag(pole) / (2 * np.pi * df))
    A = np.abs(Hs[:, :, pole_index] * np.real(pole)) * np.sign(
        np.real(Hs[:, :, pole_index])
    )
    u, s, v = np.linalg.svd(A)
    new_channel_amplitudes = v[:, 0]
    q = s[0]
    new_strings_amplitudes = u[0]
    q = (
        q
        * np.max(np.abs(new_channel_amplitudes))
        * np.max(np.abs(new_strings_amplitudes))
    )
    new_channel_amplitudes /= np.max(np.abs(new_channel_amplitudes))
    new_strings_amplitudes /= np.max(np.abs(new_strings_amplitudes))
    m = 1 / (2 * q * np.imag(pole))
    channel_amplitudes = np.vstack((channel_amplitudes, new_channel_amplitudes))
    strings_amplitudes = np.vstack((strings_amplitudes, new_strings_amplitudes))
    masses = np.append(masses, m)


file_channel_amplitudes = np.zeros((16, 128))
file_strings_amplitudes = np.zeros((44, 256))


for row, channel in enumerate(channels):
    if row == 2:
        continue
    if row > 2:
        row -= 1
    file_channel_amplitudes[channel, : len(poles)] = channel_amplitudes[:, row]

for row, string in enumerate(strings):
    if string % 2 == 0:
        file_strings_amplitudes[string // 2, : len(poles)] = strings_amplitudes[:, row]
    else:
        file_strings_amplitudes[string // 2, 128 : 128 + len(poles)] = (
            strings_amplitudes[:, row]
        )

np.savetxt(
    os.path.join(parameters_dir, "channels.txt"), file_channel_amplitudes, "%.6f", "\t"
)
np.savetxt(
    os.path.join(parameters_dir, "strings.txt"), file_strings_amplitudes, "%.6f", "\t"
)
np.savetxt(os.path.join(parameters_dir, "masses.txt"), masses, "%.12f", "\n")
np.savetxt(os.path.join(parameters_dir, "modes.txt"), pole_freqs, "%f", "\n")
np.savetxt(os.path.join(parameters_dir, "q_factor.txt"), np.real(poles), "%.6f", "\n")
