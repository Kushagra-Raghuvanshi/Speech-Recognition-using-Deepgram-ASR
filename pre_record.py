"""Deepgram ASR"""

from deepgram import Deepgram
import asyncio
import json
import csv
import pandas as pd
from functools import reduce
import os

DEEPGRAM_API_KEY = 'Insert API KEY'
directory = 'successful_calls_audio_recordings'
new_directory = 'successful_calls_audio_transcripts'

def join(series) -> str:
    return reduce(lambda x, y: x + ' ' + y, series)


async def main(audio_file_path):
    # Initializes the Deepgram SDK
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    # Open the audio file
    with open(f'{directory}/{audio_file_path}', 'rb') as audio:
        # ...or replace mimetype as appropriate
        source = {'buffer': audio, 'mimetype': 'audio/wav'}
        response = await deepgram.transcription.prerecorded(source, punctuate=True,
                                                            diarize=True,
                                                            utterances=True,
                                                            numerals=True)

        f = open(f'{new_directory}/transcript_{audio_file_path}.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(['Speaker', 'Transcript'])

        for conv_line in response["results"]["utterances"]:
            if conv_line["speaker"] == 0:
                conv_line["speaker"] = 'Bot'
            if conv_line["speaker"] == 1:
                conv_line["speaker"] = 'Caller'
            writer.writerow([conv_line["speaker"], conv_line["transcript"]])

        f.close()

        df = pd.read_csv(f'{new_directory}/transcript_{audio_file_path}.csv')
        df["cumsum"] = (df["Speaker"] != df["Speaker"].shift()).cumsum()
        aggregation_functions = {'Speaker': 'first', 'Transcript': join}
        df_new = df.groupby(df['cumsum']).aggregate(aggregation_functions)
        df_new.to_csv(f'{new_directory}/transcript_{audio_file_path}.csv')


for filename in os.listdir(directory):
    print(f"Running for {filename}")
    asyncio.run(main(filename))
