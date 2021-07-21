
def get_data(mat):
    "The (channel x times) matrix representing EEG signal in microvolt."
    segment_name = list(mat.keys())[-1]
    segment = mat[segment_name]
    return segment['data'][0][0]

def get_data_length_sec(mat):
    "The time duration of each EEG data row."
    segment_name = list(mat.keys())[-1]
    segment = mat[segment_name]
    return segment['data_length_sec'][0][0][0][0]

def get_sampling_frequency(mat):
    "The sampling frequency."
    segment_name = list(mat.keys())[-1]
    segment = mat[segment_name]
    return segment['sampling_frequency'][0][0][0][0]

def get_channels(mat):
    "List of electrode names."
    segment_name = list(mat.keys())[-1]
    segment = mat[segment_name]
    return segment['channels'][0][0][0]

def get_sequence(mat):
    "Index of data segment within the one hour series of clips."
    segment_name = list(mat.keys())[-1]
    segment = mat[segment_name]
    return segment['sequence'][0][0][0][0]
    