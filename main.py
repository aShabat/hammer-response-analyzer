import os
import re
import numpy as np
import matplotlib.pyplot as plt

from RFP import rfp_poles

# TODO
# (apply windows)??? and convert to FRF
# apply rfp
# add ui to select frequency intervals and numbers of poles???
# save to file

# Parameters (better add way for user to set them)
dir = "RESPONSES"
reference_channel = 2

# Loading responses
file_names = os.listdir(dir)
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

responses = [[[]] * len(channels) for _ in strings]
for string in range(len(strings)):
    for channel in range(len(channels)):
        file_path = os.path.join(dir, f"{strings[string]}_{channels[channel]}.txt")
        responses[string][channel] = [float(line) for line in open(file_path)]
responses = np.array(responses, dtype=np.cdouble)
references = responses[:, 2]
responses = np.delete(responses, 2, 1)

# Apply windows???

Hs = np.fft.fft(responses) / np.fft.fft(references)[:, np.newaxis]
omega = 2 * np.pi * 10 * np.arange(50000)
borders = [30, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
poles = None
for i in range(len(borders) - 1):
    new_poles = []
    for n in range(30, -2, -2):
        if n == 0:
            new_poles = []
            break
        new_poles = rfp_poles(
            2 * np.pi * np.arange(borders[i], borders[i + 1], 10),
            Hs[:, :, borders[i] // 10 : borders[i + 1] // 10],
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
    pole_index = int(np.real(pole) / (2 * np.pi * 10))
    A = np.abs(Hs[:, :, pole_index] * (-np.real(pole))) * np.sign(
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

np.savetxt("channels.txt", file_channel_amplitudes, "%.6f", "\t")
np.savetxt("strings.txt", file_strings_amplitudes, "%.6f", "\t")
np.savetxt("masses.txt", masses, "%.12f", "\n")
np.savetxt("modes.txt", pole_freqs, "%f", "\n")
