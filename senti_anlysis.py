import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the commands and their variants
commands = {
    "activate do not disturb": [
        "Please turn on do not disturb mode",
        "I need some peace and quiet, enable do not disturb",
        "Set my status to do not disturb",
        "Initiate do not disturb for me",
        "Switch on do not disturb functionality",
        "Apply do not disturb settings",
        "Activate the do not disturb feature",
        "Put me in do not disturb mode"
    ],
    "deactivate do not disturb": [
        "Turn off do not disturb",
        "Disable do not disturb mode",
        "Cancel do not disturb",
        "Deactivate the do not disturb function",
        "End do not disturb for me",
        "Remove do not disturb settings",
        "Stop do not disturb",
        "Take me out of do not disturb"
    ],
    "decline the call": [
        "Reject the incoming call",
        "Don't answer this call",
        "Send the caller to voicemail",
        "Ignore the ringing phone",
        "Hit the decline button",
        "I don't want to take this call",
        "Dismiss the current call request",
        "Let the call go to voicemail"
    ],
    "pick up the call": [
        "Answer the call",
        "Accept the incoming call",
        "Connect me to the caller",
        "Put the call through",
        "Receive the current call",
        "Let me talk to the person calling",
        "Join me to the call",
        "Patch me into the call"
    ],
    "play the music": [
        "Start playing songs",
        "Queue up some tunes",
        "Kick off the music playlist",
        "Activate music playback",
        "Get the music going",
        "Crank up the jams",
        "Fire up the audio tracks",
        "Start spinning some records"
    ],
    "pause the music": [
        "Stop the music for now",
        "Temporarily halt playback",
        "Put the music on hold",
        "Hit the pause button",
        "Suspend the current song",
        "Press pause on the audio",
        "Freeze the music track",
        "Discontinue music for a bit"
    ],
    "play the next song": [
        "Skip ahead to the next track",
        "Advance to the upcoming song",
        "Queue up the following tune",
        "Progress the playlist forward",
        "Move along to the next music file",
        "Bypass the current song",
        "Transition to the succeeding track",
        "Proceed to play the next song in line"
    ],
    "play the previous song": [
        "Go back to the prior song",
        "Rewind to the preceding track",
        "Replay the last song again",
        "Restart the previous tune",
        "Return to the song before this one",
        "Revisit the former music file",
        "Toggle back to the earlier song",
        "Jump to the preceding audio file"
    ],
    "increase the volume": [
        "Turn it up",
        "Make it louder",
        "Boost the audio level",
        "Amplify the sound",
        "Crank up the volume",
        "Raise the noise intensity",
        "Escalate the decibel levels",
        "Maximize the loudness"
    ],
    "decrease the volume": [
        "Turn it down",
        "Lower the volume",
        "Reduce the audio output",
        "Quieten the sound level",
        "Diminish the loudness",
        "Soften the noise intensity",
        "Lessen the decibel level",
        "Minimize the volume"
    ],
    "Start the vehicle": [
        "Turn on the engine",
        "Ignite the motor",
        "Fire up the vehicle",
        "Get the car going",
        "Initiate the engine startup",
        "Power on the vehicle's mechanics",
        "Activate the drive system",
        "Commence vehicle operations"
    ],
    "Stop the vehicle": [
        "Shut off the engine", 
        "Kill the motor", 
        "Power down the vehicle", 
        "Quit driving the car", 
        "Terminate vehicle operation", 
        "Deactivate the drive system", 
        "Park and turn everything off", 
        "Bring the vehicle to a full stop" 
    ]
}

# Flatten the command and variants into a list for vectorization
command_variants = []
command_labels = []

for command, variants in commands.items():
    for variant in variants:
        command_variants.append(variant)
        command_labels.append(command)

# Vectorize the command variants
vectorizer = TfidfVectorizer().fit(command_variants)
command_vectors = vectorizer.transform(command_variants)

def classify_command(user_input):
    # Preprocess and vectorize the user input
    user_vector = vectorizer.transform([user_input])

    # Compute cosine similarities
    similarities = cosine_similarity(user_vector, command_vectors)

    # Find the index of the most similar command
    most_similar_index = np.argmax(similarities)

    # Return the corresponding command
    return command_labels[most_similar_index]

def evaluate_classifier():
    correct_count = 0
    total_count = len(command_variants)

    for i, variant in enumerate(command_variants):
        predicted_command = classify_command(variant)
        if predicted_command == command_labels[i]:
            correct_count += 1

    efficacy = (correct_count / total_count) * 100
    return efficacy

# Interactive command line interface
def interactive_classification():
    print("Command Classification System")
    print("Type 'exit' to quit")

    while True:
        user_input = input("Enter your command: ")
        if user_input.lower() == 'exit':
            break
        classified_command = classify_command(user_input)
        print(f"Classified Command: {classified_command}")

# Evaluate the classifier's efficacy
print(f"Efficacy Score: {evaluate_classifier()}%")

# Start the interactive command line interface
interactive_classification()
