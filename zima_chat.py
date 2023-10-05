import tkinter as tk
from tkinter import ttk
from llama_cpp import ChatCompletionMessage, Llama
import whisper
import threading
import pyaudio
import wave 

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        root.title("Zima Chat")  # Set the window title to "Zima Chat"

        self.llm_model_path = "tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
        self.llm = Llama(model_path=self.llm_model_path, n_ctx=2048)
        self.whisper_model = whisper.load_model("base.en")
        
        self.create_interface()
        self.messages = [
            ChatCompletionMessage(role='system', content='start chat'),
            ChatCompletionMessage(role='user', content='Hello')
        ]
        self.process_message()
        self.loading = False  # Flag to track loading state

        # Bind the Ctrl+Enter keyboard shortcut to send_message
        self.root.bind("<Control-Return>", self.send_message)
        
        # Initialize audio recording parameters
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 44100
        self.audio_chunk = 1024
        self.audio_output_filename = "output.wav"
        self.audio_stream = None
        self.audio_frames = []
        
    def convert_speech_to_text(self, filepath):
        result = self.whisper_model.transcribe(filepath, fp16=False)
        return result["text"]
        
    def start_recording(self):
            if not self.loading:
                # Create a new thread to handle voice recording
                threading.Thread(target=self.record_audio).start()
                self.record_button["state"] = "disabled"  # Disable the "Record Voice" button
                self.stop_record_button["state"] = "active"  # Enable the "Stop Recording" button
                
    def stop_recording(self):
        # Set the loading flag to stop the recording
        self.loading = True
        self.stop_record_button["state"] = "disabled"  # Disable the "Stop Recording" button
        self.record_button["state"] = "active"  # Enable the "Record Voice" button
        
    def record_audio(self):
        try:
            self.audio_frames = []
            audio = pyaudio.PyAudio()

            self.audio_stream = audio.open(
                format=self.audio_format,
                channels=self.audio_channels,
                rate=self.audio_rate,
                input=True,
                frames_per_buffer=self.audio_chunk,
            )

            print("Recording...")

            while not self.loading:
                data = self.audio_stream.read(self.audio_chunk)
                self.audio_frames.append(data)

            print("Recording stopped.")

            self.audio_stream.stop_stream()
            self.audio_stream.close()
            audio.terminate()

            self.save_audio()

        except Exception as e:
            print("Error during recording:", str(e))

    def save_audio(self):
        try:
            with wave.open(self.audio_output_filename, "wb") as wf:
                wf.setnchannels(self.audio_channels)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
                wf.setframerate(self.audio_rate)
                wf.writeframes(b"".join(self.audio_frames))

            print("Audio saved to:", self.audio_output_filename)

            # Reset the loading flag to allow for re-recording
            self.loading = False
            
            # Convert the saved audio to text using Whisper (or your chosen service)
            transcribed_text = self.convert_speech_to_text(self.audio_output_filename)
    
            # Send the transcribed text as a message to LLM
            # Start the loading indicator
            self.loading = True
            self.progress.start()
            # Send the transcribed text as a message to LLM
            self.messages.append(ChatCompletionMessage(role='user', content=transcribed_text))
            self.process_message()

        except Exception as e:
            print("Error while saving audio:", str(e))

    def create_interface(self):
        self.root.geometry("600x400")  # Set the initial window size
        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(expand=True, fill=tk.BOTH)  # Make frame expand with window

        self.chat_text = tk.Text(self.chat_frame, height=20, width=50)
        self.chat_text.pack(expand=True, fill=tk.BOTH)  # Make text widget expand with frame

        self.user_input = tk.Entry(self.root)
        self.user_input.pack(expand=True, fill=tk.X)  # Make the input field expand horizontally

        # Bind the Enter key to send_message
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack()

        # Create a frame to hold the "Record Voice" and "Stop Recording" buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack()

        # Create a voice recording button
        self.record_button = tk.Button(button_frame, text="Record Voice", command=self.start_recording)
        self.record_button.grid(row=0, column=0)  # Place the "Record Voice" button in row 0, column 0

        # Create a "Stop Recording" button
        self.stop_record_button = tk.Button(button_frame, text="Stop Recording", command=self.stop_recording)
        self.stop_record_button.grid(row=0, column=1)  # Place the "Stop Recording" button in row 0, column 1
        self.stop_record_button["state"] = "disabled"  # Initially disable the button

        # Create a Progressbar widget and set it to be indeterminate
        self.progress = ttk.Progressbar(self.root, mode="indeterminate", orient=tk.HORIZONTAL)
        self.progress.pack(fill=tk.X)

    def send_message(self, event=None):
        if self.loading:
            return  # Prevent sending multiple requests while loading

        user_message = self.user_input.get()
        if user_message:
            self.messages.append(ChatCompletionMessage(role='user', content=user_message))
            self.create_user_bubble(user_message)
            self.user_input.delete(0, tk.END)

            # Start the loading indicator
            self.loading = True
            self.progress.start()

            threading.Thread(target=self.process_message).start()

    def process_message(self):
        response = self.llm.create_chat_completion(
            self.messages,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            stream=False,
            stop=[],
            max_tokens=-1,
            repeat_penalty=1.1,
        )
        output = response['choices'][0]['message']['content'] + "\n"
        self.messages.append(ChatCompletionMessage(role='assistant', content=output))
        self.update_chat_display()

        # Stop the loading indicator and clear the input field
        self.progress.stop()
        self.loading = False
        self.user_input.delete(0, tk.END)  # Clear the input field

    def update_chat_display(self):
        self.chat_text.delete(1.0, tk.END)
        for message in self.messages:
            role = message["role"].capitalize()
            content = message["content"]
            self.chat_text.insert(tk.END, f"{role}: ", role)  # Highlight role in green
            self.chat_text.insert(tk.END, f"{content}\n")

    def create_user_bubble(self, message):
        role = "User"
        content = message
        self.chat_text.insert(tk.END, f"{role}: {content}\n", "user")
        self.chat_text.tag_configure("user", foreground="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
