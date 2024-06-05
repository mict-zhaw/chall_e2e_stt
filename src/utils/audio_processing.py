import os
import subprocess

from src.utils.data_utils import get_filename


def assert_timestamp_order(timestamps) -> (bool, str, str):
    """
    Checks if the given timestamps has the right order
    :param timestamps: The timestamps to be checked
    """
    if timestamps is not None:
        for i in range(len(timestamps) - 1):
            if timestamps[i] >= timestamps[i + 1]:
                raise AssertionError(f"Invalid timestamps order. Timestamps at position {i+1}: {timestamps[i]} and {i+2}: {timestamps[i+1]}")


def batch_ffmpeg_calls(commands):
    full_command = " && ".join(commands)  # Concatenate commands with '&&'
    subprocess.check_output(full_command, shell=True, stderr=subprocess.STDOUT)


def convert_to_wav(input_file: str, output_file_name: str, sample_rate: int = None, force_overwrite: bool = False,
                   timestamps: list[tuple[float, float]] = None, assert_timestamps: bool = False, min_length: float = 0.0, verbose: bool = False):
    """
    Converts an audio file to WAV format, with the option to apply timestamps and split into chunks
    :param input_file:  The path to the input audio file
    :param output_file_name:  The desired name for the output WAV file or the base name for chunked files
    :param sample_rate:  The desired sample rate for the output WAV file (optional)
    :param force_overwrite:  If True, overwrites existing output files (optional, default is False)
    :param timestamps:  A list of timestamps indicating where to split the audio into chunks (optional)
    :param assert_timestamps:  If True, verifies the correctness of timestamp ordering (optional, default is False)
    :param min_length:  The minimum length in seconds for each chunk (optional, default is 0.0)
    :return: None
    """

    assert os.path.exists(input_file), \
        "Input file does not exist"

    if assert_timestamps:
        assert_timestamp_order(timestamps)

    try:

        base_command = ['ffmpeg']

        if force_overwrite:
            base_command.append('-y')

        base_command.extend(['-i', input_file])

        if sample_rate is not None:
            base_command.extend(['-ar', str(sample_rate)])

        output_file_names = []

        # convert audio as a whole if no timestamps, only 1 or less
        if timestamps is None or len(timestamps) == 0:

            if not force_overwrite and os.path.exists(output_file_name):
                if verbose:
                    print(f"Output file {output_file_name} already exists. Skipping conversion.")
                return [output_file_name]

            command = base_command.copy()
            command.append(output_file_name)
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            output_file_names.append(output_file_name)

        # convert audio to chunks
        else:

            for i, (start_time, end_time) in enumerate(timestamps):

                if end_time - start_time < min_length:
                    if verbose:
                        print(f"Chunk {i} for {get_filename(input_file)} is shorter than the minimum length. Skipping.")
                    output_file_names.append(None)
                    continue

                if start_time >= end_time:
                    if verbose:
                        print(f"Timestamps are invalid {start_time} - {end_time}")
                    output_file_names.append(None)
                    continue

                if len(timestamps) > 0:
                    output_chunk_file_name = output_file_name.replace(".wav", f"_{i}.wav")
                else:
                    output_chunk_file_name = output_file_name

                if not force_overwrite and os.path.exists(output_chunk_file_name):
                    if verbose:
                        print(f"Output file {output_file_name} already exists. Skipping conversion.")
                    output_file_names.append(output_chunk_file_name)
                    continue

                command = base_command.copy()
                command.extend(['-ss', str(start_time), '-to', str(end_time), output_chunk_file_name])
                subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                output_file_names.append(output_chunk_file_name)

            # todo maybe apply batching but then the problem of command to long...

        return output_file_names

    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


def remove_silence(input_audio_file: str, output_audio_file: str):
    """
    Process an audio file by detecting voice activity and removing non-speech segments.

    This function takes an input audio file, performs voice activity detection, and generates
    an output audio file containing only the speech segments with silence in between them

    :param input_audio_file:  Path to the input audio file (e.g., WAV format) that you want to process.
    :param output_audio_file:  Path to the output audio file where the processed audio will be saved.
    """

    import wave
    import librosa
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="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")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    input_audio = wave.open(input_audio_file, 'rb')
    output_audio = wave.open(output_audio_file, 'wb')
    output_audio.setparams(input_audio.getparams())

    duration = librosa.get_duration(path=input_audio_file)

    # apply voice activity check
    output = pipeline(input_audio_file)

    current_position = 0
    for speech in output.get_timeline().support():

        # add silence between active sections
        output_audio.writeframes(bytes(0 for i in range(int((speech.start - current_position) * input_audio.getframerate()) * 2))) # 2 -> 16 bit

        # Read and write the active speech segment to the output audio
        input_audio.setpos(int(speech.start * input_audio.getframerate()))
        segment_data = input_audio.readframes(int((speech.end - speech.start) * input_audio.getframerate()))
        output_audio.writeframes(segment_data)

        current_position = speech.end

    # write final silence to have same length
    output_audio.writeframes(bytes(0 for i in range(int((duration - current_position) * input_audio.getframerate()) * 2))) # 2 -> 16 bit

    # closing the file write the result
    output_audio.close()

