import tkinter as tk
from tkinter import ttk, Label
import tkinter.filedialog
from tkinter import Label
from llama_cpp import ChatCompletionMessage, Llama
import whisper
import threading
import pyaudio
import wave 
import subprocess
from PIL import Image, ImageTk

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        root.title("OpenAssistant Chat")  # Set the window title to "OpenAssistant Chat"

        self.llm_model_path = "mistral-7b-openorca.Q5_K_M.gguf"
        self.llm = Llama(model_path=self.llm_model_path, n_ctx=2048)
        self.whisper_model = whisper.load_model("base.en")
        self.llava_model_path = "/Users/hasnaincheena/Desktop/Projects/zima/llava_7b_1.5/llava-model-q5_k.gguf"
        self.llava_model_config_path = "/Users/hasnaincheena/Desktop/Projects/zima/llava_7b_1.5/mmproj-model-f16.gguf"
        self.llama_cpp_path = "../llama.cpp"
        
        
        self.create_interface()
        self.messages = [
            ChatCompletionMessage(role='system', content='start chat'),
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
        self.root.geometry("800x600")  # Set the initial window size
        
        # Create a sidebar frame
        sidebar_frame = tk.Frame(self.root, width=200, bg="black")  # Change the background color to black
        sidebar_frame.pack(fill=tk.Y, side=tk.LEFT)
        
        # Create buttons for sidebar options
        button1 = tk.Button(sidebar_frame, text="New Chat", bg="black", fg="black", command=self.new_conversation)  # Change text color to black
        button1.pack(pady=10)
        
        button2 = tk.Button(sidebar_frame, text="Save Chat", bg="black", fg="black", command=self.save_chat)  # Change text color to black
        button2.pack(pady=10)

        # Create a chat frame
        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(expand=True, fill=tk.BOTH)  # Make frame expand with window

        self.chat_text = tk.Text(self.chat_frame, height=20, width=50)
        self.chat_text.pack(expand=True, fill=tk.BOTH)  # Make text widget expand with frame

        self.user_input = tk.Entry(self.root)
        self.user_input.pack(expand=True, fill=tk.X)  # Make the input field expand horizontally

        # Bind the Enter key to send_message
        self.user_input.bind("<Return>", self.send_message)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        
        self.send_button = tk.Button(button_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=0, columnspan=1)  # Place the "Record Voice" button in row 0, column 0
        
        # Create a button for uploading images
        self.upload_image_button = tk.Button(button_frame, text="Upload Image", command=self.upload_image)
        self.upload_image_button.grid(row=0, column=1)

        # Create a voice recording button
        self.record_button = tk.Button(button_frame, text="Record Voice", command=self.start_recording)
        self.record_button.grid(row=1, column=0)  # Place the "Record Voice" button in row 0, column 0

        # Create a "Stop Recording" button
        self.stop_record_button = tk.Button(button_frame, text="Stop Recording", command=self.stop_recording)
        self.stop_record_button.grid(row=1, column=1)  # Place the "Stop Recording" button in row 0, column 1
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
        
    def new_conversation(self):
        # Reset the chat history
        self.messages = [
            ChatCompletionMessage(role='system', content='start chat'),
        ]
        self.process_message()

    def save_chat(self):
        # Prompt the user for a file name and location to save the chat
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])

        if file_path:
            try:
                with open(file_path, "w") as file:
                    for message in self.messages:
                        file.write(f"{message['role'].capitalize()}: {message['content']}\n")

                print("Chat saved to:", file_path)

            except Exception as e:
                print("Error while saving chat:", str(e))

    def load_chat_history(self):
        # Load chat history from a file if it exists
        try:
            file_path = tk.filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if file_path:
                with open(file_path, "r") as file:
                    lines = file.readlines()

                # Parse lines to create ChatCompletionMessage objects
                for line in lines:
                    parts = line.strip().split(": ")
                    role, content = parts[0], parts[1]
                    self.messages.append(ChatCompletionMessage(role=role, content=content))

                self.update_chat_display()
                print("Chat history loaded from:", file_path)

        except Exception as e:
            print("Error while loading chat history:", str(e))

    def upload_image(self):
        # Open a file dialog for image selection
        image_path = tk.filedialog.askopenfilename()
        if image_path:
            # Process the image path
            self.process_image(image_path)
    
    def process_image(self, image_path):
        # Construct the llava command with the provided image path
        llava_command = f"./llava -m {self.llava_model_path} --mmproj {self.llava_model_config_path} --image {image_path} --temp 0.1 -ngl 1"

        # Execute the llava command, capture stdout and stderr
        try:
            output = subprocess.check_output(llava_command, shell=True, text=True, cwd=self.llama_cpp_path)

            # Display the image
            image = Image.open(image_path)
            image = image.resize((200, 200))  # Adjust the size as needed
            image = ImageTk.PhotoImage(image)

            self.messages.append(ChatCompletionMessage(role='user', content=f"Image submitted: {image_path}"))
            
            # Create a Label widget to display the image
            image_label = Label(self.chat_frame, image=image)
            image_label.image = image  # Keep a reference to the image to prevent it from being garbage collected
            image_label.pack()  # Add the label to your chat frame

            # self.messages.append(ChatCompletionMessage(role='assistant', content="Describe the image in detail"))

            # Only display relevant information from stdout
            
            # Output is full of junk that we dont want to show the user
            # Trim output
            output_chunks = output.split("\n")
            # remove blank chunks
            output_chunks = [chunk for chunk in output_chunks if chunk.strip() != '']
            reply = [chunk for chunk in output_chunks if "prompt:" in output_chunks[output_chunks.index(chunk) - 1]][0].strip()
            self.messages.append(ChatCompletionMessage(role='assistant', content=reply))

            self.update_chat_display()
        except subprocess.CalledProcessError as e:
            self.messages.append(ChatCompletionMessage(role='user', content=f"Image submission failed: {str(e)}"))
            self.update_chat_display()


if __name__ == "__main__":
    root = tk.Tk()
    
    # Set icon path
    icon_path = "vacuum_cleaner_icon_250389.png"
    root.iconphoto(False, tk.PhotoImage(file=icon_path))
    
    app = ChatbotGUI(root)
    root.mainloop()
